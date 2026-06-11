"""Synthetic Multi-task TMC experiments from the paper.

This script reproduces the abstract Tree-structured Markov Chain (TMC)
simulation used for Table 3 and Appendix C figures in
"Distributional Biases in Post-Training: A Markovian Analysis of Reasoning
Trajectories".
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax, logsumexp
import math
import scipy.stats

OUTPUT_DIR = Path("figures")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the synthetic Multi-task TMC simulation.")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test with reduced budgets.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated figures.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    parser.add_argument("--num-trials", type=int, default=None, help="Override the number of question instances.")
    parser.add_argument("--finetune-iterations", type=int, default=None, help="Override fine-tuning iterations.")
    parser.add_argument("--mc-budget", type=int, default=None, help="Override reward-model Monte Carlo budget.")
    return parser.parse_args()


class Task:
    """Represents a specific reasoning task with defined start and target states"""
    def __init__(self, task_id, start_state, target_state, color='blue'):
        self.task_id = task_id
        self.start_state = start_state
        self.target_state = target_state
        self.color = color  # Color for visualization

class TMC:
    """Tree-structured Markov Chain supporting multiple tasks"""
    def __init__(self, L=3, M0=1, M=2, C_size=1, seed=42):
        np.random.seed(seed)
        self.L = L
        self.M = M
        self.C_size = C_size
        self.state_sizes = [M0] + [M] * (L-1)
        
        # State space: each layer independently numbered
        self.S = []
        for l in range(L):
            self.S.append(np.arange(self.state_sizes[l]))
        
        # Build transition kernel P(o_{l+1}|o_l) with clear edge types
        self.P = {}         # Transition probabilities
        self.edge_types = {}  # Edge types (high/low)
        self.support_set = set()  # Store all non-zero probability transitions
        
        # Define tasks
        self.tasks = {
            "TASK1": Task("TASK1", 0, 0, 'blue'),
            "TASK2": Task("TASK2", 1, 1, 'green')  # Second state in each layer
        }
        
        # Build transition kernels
        for l in range(L-1):
            for o_l in self.S[l]:
                # Randomly select high probability transition subset
                high_prob_set = set(np.random.choice(self.S[l+1], C_size, replace=False))
                prob_dist = np.zeros(len(self.S[l+1]))
                
                # Force high-probability transitions that define the two easy CoTs.
                forced_high_transitions = []
                
                # TASK1 Easy CoT: [0, 0, 0, ...]
                if o_l == 0:  # Only for state 0 in each layer
                    forced_high_transitions.append(0)  # Always transition to state 0
                
                # TASK2 Easy CoT: [1, 0, 0, ...]
                if l == 0 and o_l == 1:  # First layer, state 1
                    forced_high_transitions.append(0)  # Transition to state 0
                elif l >= 1 and o_l == 0:  # Subsequent layers, state 0
                    if l == L-2:  # Last transition layer
                        forced_high_transitions.append(1)  # Transition to state 1 for TASK2
                    else:
                        forced_high_transitions.append(0)  # Transition to state 0
                
                # Add forced transitions to high_prob_set
                for transition in forced_high_transitions:
                    if transition not in high_prob_set:
                        # Remove a random transition to make space
                        if len(high_prob_set) >= C_size:
                            # Find candidates to remove (those not in forced_high_transitions)
                            remove_candidates = list(high_prob_set - set(forced_high_transitions))
                            if remove_candidates:
                                to_remove = np.random.choice(remove_candidates)
                                high_prob_set.remove(to_remove)
                        high_prob_set.add(transition)
                for idx, o_l1 in enumerate(self.S[l+1]):
                    edge_key = (l, o_l, o_l1)
                    if o_l1 in high_prob_set:
                        prob_dist[idx] = 1.0 / C_size
                        self.edge_types[edge_key] = "high"
                    else:
                        prob_dist[idx] = 0.1
                        self.edge_types[edge_key] = "low"
                    
                    # Add to support set
                    if prob_dist[idx] > 0:
                        self.support_set.add((l, o_l, o_l1))
                
                # Normalize and store
                prob_dist /= prob_dist.sum()
                self.P[(l, o_l)] = prob_dist
    
    def get_edge_type(self, l, o_l, o_l1):
        """Get edge type (high/low)"""
        return self.edge_types.get((l, o_l, o_l1), "unknown")
    
    def is_in_support(self, l, o_l, o_l1):
        """Check if transition is in support set"""
        return (l, o_l, o_l1) in self.support_set
    
    def path_probability(self, path):
        """Calculate unnormalized path probability P_M(o)"""
        prob = 1.0
        for l in range(len(path)-1):
            o_l = path[l]
            o_l1 = path[l+1]
            prob *= self.P[(l, o_l)][o_l1]
        return prob
    
    def expected_accuracy(self, path, task_id="TASK1"):
        """
        Calculate path's expected accuracy for a specific task
        """
        task = self.tasks[task_id]
        # Verify path validity: start and end must match task
        if path[0] != task.start_state or path[-1] != task.target_state:
            return 0.0
        
        # Generate all valid paths for this task
        all_paths = self._generate_all_valid_paths(task_id)
        
        # Calculate all paths' unnormalized probabilities
        path_probs = [self.path_probability(p) for p in all_paths]
        
        # Calculate current path's probability ratio
        current_prob = self.path_probability(path)
        total_prob = sum(path_probs)
        
        return current_prob / total_prob if total_prob > 0 else 0.0
    
    def _generate_all_valid_paths(self, task_id="TASK1"):
        """
        Generate all valid paths from start_state to target_state for a task
        """
        task = self.tasks[task_id]
        
        def generate_paths_recursive(current_path, current_layer):
            if current_layer == self.L - 1:
                # Reached last layer, check if reached target state
                if current_path[-1] == task.target_state:
                    return [current_path]
                return []
            
            paths = []
            current_state = current_path[-1]
            
            # Get all possible next states
            for next_state in self.S[current_layer + 1]:
                # Check if transition is in support set
                if (current_layer, current_state, next_state) in self.support_set:
                    new_path = current_path + [next_state]
                    paths.extend(generate_paths_recursive(new_path, current_layer + 1))
            
            return paths
        
        # Generate paths starting from task's start state
        return generate_paths_recursive([task.start_state], 0)
    
    def is_valid_path_for_task(self, path, task_id="TASK1"):
        """Check if path is valid for a specific task"""
        task = self.tasks[task_id]
        return path[0] == task.start_state and path[-1] == task.target_state
    
    def is_easy_path(self, path):
        """Check if path is 'easy' (all high probability edges)"""
        for l in range(len(path)-1):
            edge_type = self.get_edge_type(l, path[l], path[l+1])
            if edge_type != "high":
                return False
        return True
    
    def print_tmc_info(self):
        """Print detailed TMC structure information"""
        print(f"TMC Structure Information:")
        print(f"- Number of layers: {self.L}")
        print(f"- State space sizes: {self.state_sizes}")
        print(f"- High probability subset size: {self.C_size}")
        print(f"- Support set size: {len(self.support_set)}")
        
        # Print task information
        for task_id, task in self.tasks.items():
            print(f"\nTask {task_id}:")
            print(f"  Start state: {task.start_state}")
            print(f"  Target state: {task.target_state}")
            
            # Generate and print valid paths
            valid_paths = self._generate_all_valid_paths(task_id)
            print(f"  Number of valid paths: {len(valid_paths)}")
            for i, path in enumerate(valid_paths):
                path_prob = self.path_probability(path)
                expected_acc = self.expected_accuracy(path, task_id)
                path_type = "EASY" if self.is_easy_path(path) else "HARD"
                print(f"    Path {i+1}: {path} (Type: {path_type}, Probability: {path_prob:.6f}, Expected Accuracy: {expected_acc:.6f})")

    def sample_path(self):
        """Sample a random path"""
        path = [np.random.choice(self.S[0])]  # Initial state
        
        for l in range(self.L-1):
            current_state = path[-1]
            prob_dist = self.P[(l, current_state)]
            next_state = np.random.choice(self.S[l+1], p=prob_dist)
            path.append(next_state)
        
        return path
    
    def sample_path_from_state(self, current_state, current_layer):
        """
        Sample complete path from specified state and layer
        Args:
            current_state: Current state
            current_layer: Current layer (0-based)
        Returns:
            Complete path starting from current state
        """
        path = [current_state]
        
        for l in range(current_layer, self.L-1):
            current_state = path[-1]
            prob_dist = self.P[(l, current_state)]
            next_state = np.random.choice(self.S[l+1], p=prob_dist)
            path.append(next_state)
        
        return path

class LinearSoftmaxModel:
    def __init__(self, state_sizes):
        self.state_sizes = state_sizes
        self.L = len(state_sizes)  # Total number of layers
        
        # Dynamically create model parameters: theta1, theta2, ..., theta_{L-1}
        self.thetas = {}
        for l in range(self.L - 1):
            # theta_l: Transition matrix from layer l to l+1
            self.thetas[l] = np.random.randn(state_sizes[l+1], state_sizes[l]) * 0.1
        
        # Record convergence metrics history
        self.sup_error_history = []  # Support set max absolute error history
        self.pass_at_k_history = []  # pass@K performance history
        self.finetune_history = []   # Finetune method-specific metrics
    
    def predict_proba(self, l, o_l):
        """Predict transition probability P(·|o_l)"""
        if l >= self.L - 1:
            raise ValueError(f"Invalid layer: {l}, max layer is {self.L-2}")
        
        # Use corresponding theta parameter
        theta = self.thetas[l]
        logits = theta @ np.eye(theta.shape[1])[:, o_l]
        return softmax(logits)
    
    def log_prob(self, l, o_l, o_l1):
        """Calculate transition log probability log P(o_l1|o_l)"""
        if l >= self.L - 1:
            raise ValueError(f"Invalid layer: {l}, max layer is {self.L-2}")
        
        # Use corresponding theta parameter
        theta = self.thetas[l]
        logits = theta @ np.eye(theta.shape[1])[:, o_l]
        log_probs = logits - logsumexp(logits)
        return log_probs[o_l1]
    
    def cross_entropy_loss(self, l, o_l, o_l1):
        """Calculate cross-entropy loss"""
        probs = self.predict_proba(l, o_l)
        return -np.log(probs[o_l1] + 1e-8)
    
    def compute_kl_divergence(self, tm):
        """
        Calculate KL divergence between model and TMC on all state transitions
        KL(P_TMC || P_model) = Σ P_TMC(i) * log(P_TMC(i) / P_model(i))
        """
        total_kl = 0.0
        
        # Calculate KL divergence for all layers
        for l in range(tm.L - 1):
            for o_l in tm.S[l]:
                # TMC true probability distribution
                p_tmc = tm.P[(l, o_l)]
                # Model predicted probability distribution
                p_model = self.predict_proba(l, o_l)
                
                # Calculate KL divergence
                kl = scipy.stats.entropy(p_tmc, p_model)
                total_kl += kl
        
        return total_kl
    
    def compute_sup_error(self, tm):
        """
        Calculate the max absolute transition-probability error on the
        nonzero TMC support.
        """
        max_error = 0.0
        
        # Iterate over all transitions in support set
        for (l, o_l, o_l1) in tm.support_set:
            # TMC true probability
            true_prob = tm.P[(l, o_l)][o_l1]
            # Model predicted probability
            pred_prob = self.predict_proba(l, o_l)[o_l1]
            
            # Calculate absolute error
            error = abs(pred_prob - true_prob)
            if error > max_error:
                max_error = error
        
        return max_error
    
    def compute_pass_at_k(self, tm, task_id="TASK1", K=3, num_trials=100):
        """
        Calculate pass@K performance for a specific task
        Proportion of trials where at least one correct answer is found in K parallel experiments
        """
        task = tm.tasks[task_id]
        correct_count = 0
        
        for _ in range(num_trials):
            # Sample K paths
            paths = [self.sample_path(task.start_state) for _ in range(K)]
            
            # Check if at least one path is correct for this task
            for path in paths:
                if path[-1] == task.target_state:
                    correct_count += 1
                    break
        
        return correct_count / num_trials
    
    def sample_path(self, start_state):
        """Sample path from given start state"""
        path = [start_state]
        state = start_state
        current_layer = 0
        
        while current_layer < self.L - 1:
            # Predict next state probability
            probs = self.predict_proba(current_layer, state)
            next_state = np.random.choice(range(len(probs)), p=probs)
            path.append(next_state)
            state = next_state
            current_layer += 1
        
        return path
    
    def train(self, tm, T1=1000, T2=500, eta=0.001, eval_interval=100):
        """Algorithm 1: Two-phase training strategy"""
        # Initialize history records
        self.sup_error_history = []
        self.pass_at_k_history = []
        
        # Record initial KL divergence
        initial_kl = self.compute_kl_divergence(tm)
        print(f"Initial KL Divergence: {initial_kl:.6f}")
        
        # Phase 1 training
        for t in range(T1):
            path = tm.sample_path()
            
            # Gradient update for each layer
            for l in range(len(path)-1):
                o_l = path[l]
                o_l1 = path[l+1]
                
                # Create target vector
                target = np.zeros(self.state_sizes[l+1])
                target[o_l1] = 1
                
                # Predict probabilities
                probs = self.predict_proba(l, o_l)
                
                # Calculate gradient
                if l < self.L - 1:  # Ensure layer is within valid range
                    grad = np.outer(probs - target, np.eye(self.thetas[l].shape[1])[:, o_l])
                    self.thetas[l] -= eta * grad
            
            # Record convergence metrics
            if t % eval_interval == 0:
                sup_error = self.compute_sup_error(tm)
                pass_at_k = self.compute_pass_at_k(tm, "TASK1", K=3)
                
                self.sup_error_history.append((t, sup_error))
                self.pass_at_k_history.append((t, pass_at_k))
        
        # Phase 2 training
        for t in range(T1, T1+T2):
            path = tm.sample_path()
            
            # Gradient update for each layer
            for l in range(len(path)-1):
                o_l = path[l]
                o_l1 = path[l+1]
                
                # Check if probability is too low
                probs = self.predict_proba(l, o_l)
                if np.max(probs) > 1e-6:
                    # Create target vector
                    target = np.zeros(self.state_sizes[l+1])
                    target[o_l1] = 1
                    
                    # Calculate gradient
                    if l < self.L - 1:  # Ensure layer is within valid range
                        grad = np.outer(probs - target, np.eye(self.thetas[l].shape[1])[:, o_l])
                        self.thetas[l] -= eta * grad * 0.5
            
            # Record convergence metrics
            if t % eval_interval == 0:
                sup_error = self.compute_sup_error(tm)
                pass_at_k = self.compute_pass_at_k(tm, "TASK1", K=3)
                
                self.sup_error_history.append((t, sup_error))
                self.pass_at_k_history.append((t, pass_at_k))
        
        # Record final KL divergence
        final_kl = self.compute_kl_divergence(tm)
        print(f"Final KL Divergence: {final_kl:.6f}")
        
        # Print max absolute error table in markdown format
        print("\nMax Absolute Error History:")
        print("| Iteration | Max Absolute Error |")
        print("|-----------|---------------------|")
        for t, error in self.sup_error_history:
            print(f"| {t} | {error:.6f} |")

    # ================== Finetune Methods ====================
    def finetune(self, tm, reward_model, finetune_question_instances, method="REINFORCE",
                 iterations=1000, lr=0.01, beta=1.0, epsilon_clip=0.2, N=10,
                 rej_threshold=5, eval_k=30, progress_interval=100):
        """
        Fine-tune the pretrained model using various RL algorithms
        """
        print(f"\nStarting {method} finetuning for {iterations} iterations...")
        self.finetune_history = []
        
        # Store old model parameters for PPO updates
        if method in ["PPO"]:
            old_thetas = {l: np.copy(self.thetas[l]) for l in self.thetas}
        
        for iter in range(iterations):
            # Sample a question instance (TASK1)
            q_idx = np.random.randint(len(finetune_question_instances))
            correct_cots = finetune_question_instances[q_idx]
            
            if method == "REINFORCE":
                self._reinforce_update(tm, reward_model, correct_cots, lr, N)
            elif method == "RAFT":
                self._raft_update(tm, reward_model, correct_cots, lr, N)
            elif method == "PPO":
                self._ppo_update(tm, reward_model, correct_cots, old_thetas, lr, epsilon_clip, N)
                old_thetas = {l: np.copy(self.thetas[l]) for l in self.thetas}
            elif method == "Reinforce-rej":
                self._reinforce_rej_update(tm, reward_model, correct_cots, lr, beta, N, rej_threshold)
            else:
                raise ValueError(f"Unknown finetune method: {method}")
            
            # Record performance periodically
            if iter % progress_interval == 0:
                # Evaluate on TASK1 with question_instances
                pass_at_k = self._evaluate_pass_at_k(tm, finetune_question_instances, K=eval_k)
                self.finetune_history.append((iter, pass_at_k))
                print(f"Iteration {iter}/{iterations} - Pass@K: {pass_at_k:.4f}")
        
        print(f"{method} finetuning completed. Final Pass@K: {self.finetune_history[-1][1]:.4f}")
    
    def _reinforce_update(self, tm, reward_model, correct_cots, lr, N):
        """REINFORCE algorithm update with 0-1 reward signal"""
        # Sample path and calculate reward
        path = self.sample_path(tm.tasks["TASK1"].start_state)
        
        # 0-1 reward: 1 if path is in correct CoT set, else 0
        R = 1 if any(tuple(path) == tuple(cot) for cot in correct_cots) else 0
        
        # Gradient update for each transition
        for l in range(len(path)-1):
            o_l = path[l]
            o_l1 = path[l+1]
            
            # Compute gradient of log probability: ∇logπ(a|s)
            grad_logp = self._compute_grad_logp(l, o_l, o_l1)
            
            # Update parameters: θ = θ + lr * R * ∇logπ(a|s)
            self.thetas[l] += lr * R * grad_logp
    
    def _raft_update(self, tm, reward_model, correct_cots, lr, N):
        """RAFT algorithm update with 0-1 reward signal"""
        # Sample path and calculate reward
        path = self.sample_path(tm.tasks["TASK1"].start_state)
        
        # 0-1 reward: 1 if path is in correct CoT set, else 0
        R = 1 if any(tuple(path) == tuple(cot) for cot in correct_cots) else 0
        
        # Compute total reward for RAFT
        total_R = 0
        log_probs = []
        for l in range(len(path)-1):
            log_p = self.log_prob(l, path[l], path[l+1])
            log_probs.append(log_p)
            total_R += log_p * R
        
        # Gradient update for each transition
        for l in range(len(path)-1):
            o_l = path[l]
            o_l1 = path[l+1]
            
            # Compute gradient of log probability: ∇logπ(a|s)
            grad_logp = self._compute_grad_logp(l, o_l, o_l1)
            
            # Compute RAFT-specific multiplier: (1 + log_p)
            term = (1 + log_probs[l])
            
            # Update parameters: θ = θ + lr * term * ∇logπ(a|s) * R
            self.thetas[l] += lr * term * grad_logp * R
    
    def _ppo_update(self, tm, reward_model, correct_cots, old_thetas, lr, epsilon_clip, N):
        """PPO algorithm update with 0-1 reward signal"""
        # Sample path using OLD parameters
        path = self.sample_path(tm.tasks["TASK1"].start_state)
        
        # 0-1 reward: 1 if path is in correct CoT set, else 0
        R = 1 if any(tuple(path) == tuple(cot) for cot in correct_cots) else 0
        
        # Compute advantage function
        advantages = self._compute_advantages(path, tm, reward_model, N)
        
        # Gradient update for each transition
        for l in range(len(path)-1):
            o_l = path[l]
            o_l1 = path[l+1]
            
            # Current policy probability
            current_prob = self.predict_proba(l, o_l)[o_l1]
            
            # Old policy probability (from before update)
            old_theta = old_thetas[l]
            logits = old_theta @ np.eye(old_theta.shape[1])[:, o_l]
            old_probs = softmax(logits)
            old_prob = old_probs[o_l1]
            
            # Probability ratio: r = π_new / π_old
            r = current_prob / (old_prob + 1e-8)
            
            # Compute gradient of log probability: ∇logπ(a|s)
            grad_logp = self._compute_grad_logp(l, o_l, o_l1)
            
            # PPO clipped objective
            advantage = advantages[l]
            sign = 2 * (advantage >= 0) - 1  # 1 if adv >=0, else -1
            clip_term = sign * epsilon_clip
            
            # Update parameters: θ = θ + lr * advantage * (1 ± ε_clip) * ∇logπ(a|s)
            self.thetas[l] += lr * (1 + clip_term) * advantage * grad_logp
    
    def _reinforce_rej_update(self, tm, reward_model, correct_cots, lr, beta, N, rej_threshold):
        """Reinforce-rejection algorithm with threshold-based rejection"""
        # Check if pretrained model can solve this question with rej_threshold trials
        pretrained_model = self  # We use current model as pretrained for simplicity
        solved = False
        
        for _ in range(rej_threshold):
            path = pretrained_model.sample_path(tm.tasks["TASK1"].start_state)
            if any(tuple(path) == tuple(cot) for cot in correct_cots):
                solved = True
                break
        
        # Only update if pretrained model cannot solve the question
        if not solved:
            # Sample path and calculate reward
            path = self.sample_path(tm.tasks["TASK1"].start_state)
            
            # 0-1 reward: 1 if path is in correct CoT set, else 0
            R = 1 if any(tuple(path) == tuple(cot) for cot in correct_cots) else 0
            
            # Gradient update for each transition
            for l in range(len(path)-1):
                o_l = path[l]
                o_l1 = path[l+1]
                
                # Compute gradient of log probability: ∇logπ(a|s)
                grad_logp = self._compute_grad_logp(l, o_l, o_l1)
                
                # Update parameters: θ = θ + lr * R * ∇logπ(a|s)
                self.thetas[l] += lr * R * grad_logp
    
    def _compute_grad_logp(self, l, o_l, o_l1):
        """Compute gradient of log probability w.r.t parameters"""
        # Gradient of log softmax: ∇logπ(a|s) = (I - p(·|s)) for selected action
        probs = self.predict_proba(l, o_l)
        grad_logp = -probs
        grad_logp[o_l1] = 1 - probs[o_l1]
        return np.outer(grad_logp, np.eye(self.thetas[l].shape[1])[:, o_l])
    
    def _compute_advantages(self, path, tm, reward_model, N=100):
        """
        Compute advantages using potential functions
        A_{l+1}^{k}(o_l, o_{l+1}) = Q^{k}(o_l, o_{l+1}) - V^{k}(o_l)
        """
        advantages = []
        
        # Compute potential for each state along the path
        potentials = []
        for i in range(len(path)):
            partial_path = path[:i+1]
            potentials.append(reward_model.potential_reward(partial_path, N))
        
        # Calculate advantages based on potential differences
        for i in range(len(path)-1):
            # Q(o_l, o_{l+1}) = potential of next state
            Q_value = potentials[i+1]
            
            # V(o_l) = expected Q for current state
            current_state = path[i]
            current_layer = i
            state_value = 0.0
            
            # Estimate state value by averaging over possible next states
            for next_state in tm.S[current_layer+1]:
                # Generate sample paths for potential calculation
                sample_path = tm.sample_path_from_state(next_state, current_layer+1)
                partial = path[:i] + [current_state, next_state] + sample_path[1:]
                state_value += tm.expected_accuracy(partial) * self.predict_proba(i, current_state)[next_state]
            
            # Advantage: Q(o_l, o_{l+1}) - V(o_l)
            advantages.append(Q_value - state_value)
        
        return advantages

    def _evaluate_pass_at_k(self, tm, question_instances, K=3):
        """
        Evaluate pass@K performance for TASK1 using question_instances
        """
        correct_count = 0
        
        for correct_cots in question_instances:
            # Sample K paths
            paths = [self.sample_path(tm.tasks["TASK1"].start_state) for _ in range(K)]
            
            # Check if at least one path is in correct CoT set
            solved = False
            for path in paths:
                if any(tuple(path) == tuple(cot) for cot in correct_cots):
                    solved = True
                    break
            
            if solved:
                correct_count += 1
        
        return correct_count / len(question_instances)

    def plot_convergence_metrics(self):
        """Plot convergence metric history"""
        if not self.sup_error_history:
            print("No convergence metric data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Support set max absolute error history
        iterations, sup_errors = zip(*self.sup_error_history)
        plt.plot(iterations, sup_errors)
        plt.title("Support Set Max Absolute Error")
        plt.xlabel("Iteration")
        plt.ylabel("Max Absolute Error")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_finetune_progress(self):
        """Plot finetune progress history"""
        if not self.finetune_history:
            print("No finetune progress data to plot")
            return
        
        iterations, pass_at_k = zip(*self.finetune_history)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, pass_at_k)
        plt.title("Pass@K During Finetuning")
        plt.xlabel("Iteration")
        plt.ylabel("Pass@K Rate")
        plt.grid(True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f'finetune_progress_{len(self.finetune_history)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Finetune progress plot saved as {output_path}")

class PPOSampler:
    """Implement PPO-Sampling using pretrained model"""
    def __init__(self, tm, reward_model, pretrained_model):
        self.tm = tm
        self.reward_model = reward_model
        self.pretrained_model = pretrained_model
    
    def sample_path(self, temperature=1.0, beta=1.0, task_id="TASK1"):
        """
        PPO-Sampling based on Eq.(4):
        p_θ^{PO}(o_{l+1}|o_l) ∝ p_θ*(o_{l+1}|o_l) * exp( (r_hat * A_l^k(o_{l+1})) / β )
        """
        path = [self.tm.tasks[task_id].start_state]  # Start with specified task
        
        for l in range(self.tm.L - 1):
            current_state = path[-1]
            pretrained_probs = self.pretrained_model.predict_proba(l, current_state)
            
            # Calculate advantages for each next state
            advantages = []
            for o_l1 in self.tm.S[l+1]:
                # Advantage: Q(o_l, o_{l+1}) - V(o_l)
                partial_path = path + [o_l1]
                Q_value = self.reward_model.potential_reward(partial_path, 15, task_id)  # Use N=15 and task_id
                V_value = self.reward_model.potential_reward(path, 15, task_id)  # Use N=15 and task_id
                advantages.append(Q_value - V_value)
            
            # Adjust probabilities with advantages
            adjusted_probs = pretrained_probs * np.exp(np.array(advantages) * temperature / beta)
            adjusted_probs /= adjusted_probs.sum()
            
            # Sample next state
            next_state = np.random.choice(range(len(adjusted_probs)), p=adjusted_probs)
            path.append(next_state)
        
        return path

class InferenceSampler:
    """Implement inference sampling strategies using TMC directly"""
    def __init__(self, tm, reward_model):
        self.tm = tm
        self.reward_model = reward_model
    
    def temperature_adjusted_sampling(self, num_samples=10, temperature=1.0, N=100):
        """
        Temperature-adjusted sampling based on Outcome reward
        Eq.(16): p(o|q) ∝ exp(Rex_{q,k}(o)*temperature)
        Use TMC directly for sampling
        """
        # Generate unique candidate paths.
        candidate_paths = []
        unique_paths = set()
        
        while len(candidate_paths) < N:
            path = self.tm.sample_path()
            path_tuple = tuple(path)
            if path_tuple not in unique_paths:
                candidate_paths.append(path)
                unique_paths.add(path_tuple)
        
        # Compute the outcome reward for each path.
        rewards = [self.reward_model.outcome_reward(path) for path in candidate_paths]
        
        # Sample with replacement from temperature-adjusted probabilities.
        exp_rewards = np.exp(np.array(rewards) * temperature)
        probs = exp_rewards / exp_rewards.sum()
        
        selected_indices = np.random.choice(
            len(candidate_paths), size=num_samples, p=probs, replace=True
        )
        return [candidate_paths[i] for i in selected_indices]
    
    def outcome_bon_sampling(self, num_samples=10, N=100):
        """
        Best-of-N sampling based on Outcome reward
        Use TMC directly for sampling
        """
        # Generate N candidate paths
        candidate_paths = [self.tm.sample_path() for _ in range(N)]
        
        # Calculate reward for each path
        rewards = [self.reward_model.outcome_reward(path) for path in candidate_paths]
        
        # Select highest reward num_samples paths
        sorted_indices = np.argsort(rewards)[::-1]
        selected_indices = sorted_indices[:num_samples]
        
        return [candidate_paths[i] for i in selected_indices]
    
    def potential_soft_sampling(self, partial_path=None, temperature=1.0, N=100):
        """
        Soft sampling based on Potential reward
        Use TMC directly for sampling
        """
        if partial_path is None:
            partial_path = [self.tm.tasks["TASK1"].start_state]  # TASK1 default
        
        current_layer = len(partial_path) - 1
        current_state = partial_path[-1]
        
        # Return current path if no subsequent layers
        if current_layer >= self.tm.L - 1:
            return partial_path
        
        # Get all possible next states
        possible_states = self.tm.S[current_layer + 1]
        
        # Calculate Potential reward for each possible state
        rewards = []
        for next_state in possible_states:
            new_partial = partial_path + [next_state]
            rewards.append(self.reward_model.potential_reward(new_partial, N))
        
        # Adjust sampling probabilities with rewards
        exp_rewards = np.exp(np.array(rewards) * temperature)
        probs = exp_rewards / exp_rewards.sum()
        
        # Sample next state
        next_state = np.random.choice(possible_states, p=probs)
        
        # Recursively generate subsequent path
        new_partial = partial_path + [next_state]
        return self.potential_soft_sampling(new_partial, temperature, N)
    
    def dprm_sampling(self, partial_path=None, temperature=1.0, N=100):
        """
        Sampling based on R_{DPRM}^k(o_l)
        Use TMC directly for sampling
        """
        if partial_path is None:
            partial_path = [self.tm.tasks["TASK1"].start_state]  # TASK1 default
        
        current_layer = len(partial_path) - 1
        current_state = partial_path[-1]
        
        # Return current path if no subsequent layers
        if current_layer >= self.tm.L - 1:
            return partial_path
        
        # Get all possible next states
        possible_states = self.tm.S[current_layer + 1]
        
        # Calculate DPRM reward for each possible state
        rewards = []
        for next_state in possible_states:
            new_partial = partial_path + [next_state]
            rewards.append(self.reward_model.DPRM(new_partial, temperature, N))
        
        # Adjust sampling probabilities with rewards
        exp_rewards = np.exp(np.array(rewards) * temperature)
        probs = exp_rewards / exp_rewards.sum()
        
        # Sample next state
        next_state = np.random.choice(possible_states, p=probs)
        
        # Recursively generate subsequent path
        new_partial = partial_path + [next_state]
        return self.dprm_sampling(new_partial, temperature, N)
    
    def potential_bon_sampling(self, partial_path=None, N=100):
        """
        Strict Best-of-N sampling based on Potential reward
        Use TMC directly for sampling
        """
        if partial_path is None:
            partial_path = [self.tm.tasks["TASK1"].start_state]  # TASK1 default
        
        current_layer = len(partial_path) - 1
        current_state = partial_path[-1]
        
        # Return current path if no subsequent layers
        if current_layer >= self.tm.L - 1:
            return partial_path
        
        # Get all possible next states
        possible_states = self.tm.S[current_layer + 1]
        
        # Calculate reward for each candidate state
        rewards = []
        for next_state in possible_states:
            new_partial = partial_path + [next_state]
            rewards.append(self.reward_model.potential_reward(new_partial, N))
        
        # Select highest reward state
        max_reward = max(rewards)
        best_states = [s for s, r in zip(possible_states, rewards) if r == max_reward]
        next_state = np.random.choice(best_states)
        
        # Recursively generate subsequent path
        new_partial = partial_path + [next_state]
        return self.potential_bon_sampling(new_partial, N)

    def dprm_bon_sampling(self, partial_path=None, N=100, temperature=1.0):
        """
        Strict Best-of-N sampling based on DPRM reward
        Use TMC directly for sampling
        """
        if partial_path is None:
            partial_path = [self.tm.tasks["TASK1"].start_state]  # TASK1 default
        
        current_layer = len(partial_path) - 1
        current_state = partial_path[-1]
        
        # Return current path if no subsequent layers
        if current_layer >= self.tm.L - 1:
            return partial_path
        
        # Get all possible next states
        possible_states = self.tm.S[current_layer + 1]
        
        # Calculate DPRM reward for each candidate state
        rewards = []
        for next_state in possible_states:
            new_partial = partial_path + [next_state]
            rewards.append(self.reward_model.DPRM(new_partial, temperature, N))
        
        # Select highest reward state
        max_reward = max(rewards)
        best_states = [s for s, r in zip(possible_states, rewards) if r == max_reward]
        next_state = np.random.choice(best_states)
        
        # Recursively generate subsequent path
        new_partial = partial_path + [next_state]
        return self.dprm_bon_sampling(new_partial, N, temperature)

class RewardModel:
    """Implement three reward models: ORM, PRM and DPRM using TMC directly"""
    def __init__(self, tm, mc_budget=100):
        self.tm = tm
        self.mc_budget = mc_budget  # Monte Carlo sampling budget
        self._potential_cache = {}
        self._dprm_cache = {}
    
    def outcome_reward(self, full_path, task_id="TASK1"):
        """
        Rex_{q,k}(o): Outcome Reward Model (accuracy)
        Implement Definition D.1 Eq.(33)
        """
        return self.tm.expected_accuracy(full_path, task_id)
    
    def potential_reward(self, partial_path, N=None, task_id="TASK1"):
        """
        R_{potential}^k(o_l): Process Reward Model (original version)
        Eq.(14), defined as expected accuracy of current state in future
        Use fixed budget N for Monte Carlo estimation
        """
        if not partial_path:
            return 0.0
        num_samples = N if N is not None else self.mc_budget
        cache_key = (tuple(partial_path), int(num_samples), task_id)
        if cache_key in self._potential_cache:
            return self._potential_cache[cache_key]
        
        # Extract current state and layer index
        l = len(partial_path) - 1
        current_state = partial_path[-1]
        
        # Use Monte Carlo sampling to get complete paths
        total_reward = 0.0
        
        for _ in range(num_samples):
            # Sample complete path from current state
            path = self.tm.sample_path_from_state(current_state, l)
            full_path = partial_path + path[1:]  # Combine partial path and subsequent path
            total_reward += self.outcome_reward(full_path, task_id)
        
        value = total_reward / num_samples
        self._potential_cache[cache_key] = value
        return value
    
    def DPRM(self, partial_path, temperature=1.0, N=None, task_id="TASK1"):
        """
        R_{DPRM}^k(o_l): Disentangled Process Reward Model
        Definition 4.3, Eq.(18)
        Use single temperature parameter to control reward calculation
        """
        if not partial_path:
            return 0.0
        num_samples = N if N is not None else self.mc_budget
        cache_key = (tuple(partial_path), float(temperature), int(num_samples), task_id)
        if cache_key in self._dprm_cache:
            return self._dprm_cache[cache_key]
        l = len(partial_path) - 1
        current_state = partial_path[-1]
        
        # Monte Carlo sampling
        exp_rewards = []
        
        for _ in range(num_samples):
            # Sample complete path from current state
            path = self.tm.sample_path_from_state(current_state, l)
            full_path = partial_path + path[1:]
            reward = self.outcome_reward(full_path, task_id)
            exp_rewards.append(math.exp(temperature * reward))
        
        # Calculate expectation: E[exp(λ Rex)] = (1/N) * Σ exp(λ * r)
        if exp_rewards:
            expectation = sum(exp_rewards) / num_samples
            # DPRM reward: R^k_DPRM(o_l) = (1/λ) * log h_k(o_l)
            value = math.log(expectation) / temperature
            self._dprm_cache[cache_key] = value
            return value
        return 0.0

def plot_task_performance(tm, reward_model, task_id, question_instances, 
                          linear_model=None, finetuned_models=None, ppo_sampler=None,
                          K=3, N=100, num_trials=100, temperature=0.5,
                          show_plot=True):
    """
    Plot pass@K performance for a specific task and print results
    """
    # Create sampler
    sampler = InferenceSampler(tm, reward_model)
    
    # Define sampling strategies with correct order and names
    strategies = [
        ("Base Model", lambda: [tm.sample_path() for _ in range(K)]),
    ]
    
    # Add Finetune methods if provided (in correct order)
    if finetuned_models:
        # Add REINFORCE
        if "REINFORCE" in finetuned_models:
            strategies.append(("REINFORCE", lambda: [finetuned_models["REINFORCE"].sample_path(tm.tasks[task_id].start_state) for _ in range(K)]))
        
        # Add RAFT
        if "RAFT" in finetuned_models:
            strategies.append(("RAFT", lambda: [finetuned_models["RAFT"].sample_path(tm.tasks[task_id].start_state) for _ in range(K)]))
        
        # Add PPO
        if "PPO" in finetuned_models:
            strategies.append(("PPO", lambda: [finetuned_models["PPO"].sample_path(tm.tasks[task_id].start_state) for _ in range(K)]))
        
        # Add Reinforce-rej
        if "Reinforce-rej" in finetuned_models:
            strategies.append(("Reinforce-rej", lambda: [finetuned_models["Reinforce-rej"].sample_path(tm.tasks[task_id].start_state) for _ in range(K)]))
    
    # Add PPO-Sampling (GRPO-KL)
    if ppo_sampler:
        strategies.append(("GRPO-KL", lambda: [ppo_sampler.sample_path(temperature=temperature) for _ in range(K)]))
    
    # Add inference methods
    strategies.extend([
        ("ORM-TAS", lambda: sampler.temperature_adjusted_sampling(num_samples=K, temperature=temperature, N=N)),
        ("ORM-BoN", lambda: sampler.outcome_bon_sampling(num_samples=K, N=N)),
        ("PRM-BoN", lambda: [sampler.potential_bon_sampling(partial_path=[tm.tasks[task_id].start_state], N=N) for _ in range(K)]),
        ("DPRM", lambda: [sampler.dprm_sampling(partial_path=[tm.tasks[task_id].start_state], temperature=temperature, N=N) for _ in range(K)]),
    ])
    
    # Calculate pass@K performance for each strategy
    performance = []
    strategy_names = []
    
    for name, sample_func in strategies:
        total_success = 0
        
        for question_idx, correct_cots in enumerate(question_instances):
            # For each Question Instance, perform K parallel samples
            question_success = False
            
            # Sample K paths in parallel
            paths = sample_func()
            
            # Check if at least one path solves this Question
            for path in paths:
                if any(tuple(path) == tuple(cot) for cot in correct_cots):
                    question_success = True
                    break
            
            if question_success:
                total_success += 1
        
        # pass@K = number of solved Questions / total Questions
        pass_rate = total_success / num_trials
        performance.append(pass_rate)
        strategy_names.append(name)
    
    # Print results in markdown-friendly format with percentages
    print(f"\nPass@{K} Performance for {task_id}:")
    print("| Strategy | Pass@K Rate (%) |")
    print("|----------|-----------------|")
    for i, name in enumerate(strategy_names):
        percentage = performance[i] * 100
        print(f"| {name} | {percentage:.2f}% |")
    
    # Plot performance
    plt.figure(figsize=(14, 8))
    plt.bar(strategy_names, performance, color=tm.tasks[task_id].color)
    plt.title(f"Pass@{K} Performance for {task_id}")
    plt.xlabel("Sampling Strategy")
    plt.ylabel(f"Pass@{K} Rate")
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        # Save plot instead of showing to avoid blocking
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f'{task_id}_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{task_id} performance plot saved as {output_path}")

def plot_task_coverage(tm, reward_model, task_id, question_instances, 
                        linear_model=None, finetuned_models=None, ppo_sampler=None,
                        K=3, N=100, num_trials=100, temperature=0.5,
                        show_plot=True):
    """
    Plot coverage of valid CoTs for a specific task
    """
    # Create sampler
    sampler = InferenceSampler(tm, reward_model)
    
    # Define sampling strategies with correct order and names
    strategies = [
        ("Base Model", lambda: [tm.sample_path() for _ in range(K)]),
    ]
    
    # Add Finetune methods if provided (in correct order)
    if finetuned_models:
        # Add REINFORCE
        if "REINFORCE" in finetuned_models:
            strategies.append(("REINFORCE", lambda: [finetuned_models["REINFORCE"].sample_path(tm.tasks[task_id].start_state) for _ in range(K)]))
        
        # Add RAFT
        if "RAFT" in finetuned_models:
            strategies.append(("RAFT", lambda: [finetuned_models["RAFT"].sample_path(tm.tasks[task_id].start_state) for _ in range(K)]))
        
        # Add PPO
        if "PPO" in finetuned_models:
            strategies.append(("PPO", lambda: [finetuned_models["PPO"].sample_path(tm.tasks[task_id].start_state) for _ in range(K)]))
        
        # Add Reinforce-rej
        if "Reinforce-rej" in finetuned_models:
            strategies.append(("Reinforce-rej", lambda: [finetuned_models["Reinforce-rej"].sample_path(tm.tasks[task_id].start_state) for _ in range(K)]))
    
    # Add PPO-Sampling (GRPO-KL)
    if ppo_sampler:
        strategies.append(("GRPO-KL", lambda: [ppo_sampler.sample_path(temperature=temperature) for _ in range(K)]))
    
    # Add inference methods
    strategies.extend([
        ("ORM-TAS", lambda: sampler.temperature_adjusted_sampling(num_samples=K, temperature=temperature, N=N)),
        ("ORM-BoN", lambda: sampler.outcome_bon_sampling(num_samples=K, N=N)),
        ("PRM-BoN", lambda: [sampler.potential_bon_sampling(partial_path=[tm.tasks[task_id].start_state], N=N) for _ in range(K)]),
        ("DPRM", lambda: [sampler.dprm_sampling(partial_path=[tm.tasks[task_id].start_state], temperature=temperature, N=N) for _ in range(K)]),
    ])
    
    # For each strategy, track coverage of valid CoTs
    strategy_names = []
    easy_proportions = []
    hard_proportions = []
    invalid_proportions = []    
    
    for name, sample_func in strategies:
        easy_count = 0
        hard_count = 0
        invalid_count = 0
        total_samples = 0
        
        for _ in range(num_trials):
            # For each trial, perform K parallel samples
            paths = sample_func()
            
            for path in paths:
                total_samples += 1
                if tm.is_valid_path_for_task(path, task_id):
                    if tm.is_easy_path(path):
                        easy_count += 1
                    else:
                        hard_count += 1
                else:
                    invalid_count += 1
        
        # Calculate proportions
        easy_prop = easy_count / total_samples
        hard_prop = hard_count / total_samples
        invalid_prop = invalid_count / total_samples
        
        strategy_names.append(name)
        easy_proportions.append(easy_prop)
        hard_proportions.append(hard_prop)
        invalid_proportions.append(invalid_prop)
    
    # Print results in markdown-friendly format with percentages
    print(f"\n{task_id} Valid CoT Coverage:")
    print("| Strategy | Easy Valid (%) | Hard Valid (%) | Invalid (%) |")
    print("|----------|----------------|----------------|-------------|")
    for i, name in enumerate(strategy_names):
        easy_pct = easy_proportions[i] * 100
        hard_pct = hard_proportions[i] * 100
        invalid_pct = invalid_proportions[i] * 100
        print(f"| {name} | {easy_pct:.2f}% | {hard_pct:.2f}% | {invalid_pct:.2f}% |")
    
    # Plot stacked bar chart
    plt.figure(figsize=(16, 10))
    
    # Plot invalid proportion (gray)
    p1 = plt.bar(strategy_names, invalid_proportions, color='gray')
    
    # Plot hard valid CoTs (red)
    p2 = plt.bar(strategy_names, hard_proportions, bottom=invalid_proportions, color='red')
    
    # Plot easy valid CoTs (green)
    p3 = plt.bar(strategy_names, easy_proportions, 
                bottom=[i+h for i, h in zip(invalid_proportions, hard_proportions)], 
                color='green')
    
    plt.title(f"{task_id} Valid CoT Coverage (K={K}, N={N}, {num_trials} trials)")
    plt.xlabel("Sampling Strategy")
    plt.ylabel("Proportion of Sampled Paths")
    plt.legend((p1[0], p2[0], p3[0]), ('Invalid', 'Hard Valid', 'Easy Valid'))
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        # Save plot instead of showing to avoid blocking
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f'{task_id}_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{task_id} coverage plot saved as {output_path}")

def generate_question_instances(tm, num_instances, task_id="TASK1"):
    """
    Generate question instances with correct CoT sets for a specific task
    Each instance has a defined set of correct CoTs based on expected accuracy
    """
    question_instances = []
    for _ in range(num_instances):
        valid_cots = tm._generate_all_valid_paths(task_id)
        correct_cots = []
        
        for cot in valid_cots:
            expected_acc = tm.expected_accuracy(cot, task_id)
            if np.random.random() < expected_acc:
                correct_cots.append(cot)
        
        question_instances.append(correct_cots)
    return question_instances

def evaluate_all_tasks(tm, reward_model, task1_instances, task2_instances,
                      linear_model=None, finetuned_models=None, ppo_sampler=None,
                      K=3, N=100, num_trials=100, show_plot=True):
    """
    Unified function to evaluate both tasks for performance and coverage
    """
    # Create sampler
    sampler = InferenceSampler(tm, reward_model)
    
    # Define sampling strategies with correct order and names
    strategies = [
        ("Base Model", lambda: [tm.sample_path() for _ in range(K)]),
    ]
    
    # Add Finetune methods if provided (in correct order)
    if finetuned_models:
        # Add REINFORCE
        if "REINFORCE" in finetuned_models:
            strategies.append(("REINFORCE", lambda: [finetuned_models["REINFORCE"].sample_path(tm.tasks["TASK1"].start_state) for _ in range(K)]))
        
        # Add RAFT
        if "RAFT" in finetuned_models:
            strategies.append(("RAFT", lambda: [finetuned_models["RAFT"].sample_path(tm.tasks["TASK1"].start_state) for _ in range(K)]))
        
        # Add PPO
        if "PPO" in finetuned_models:
            strategies.append(("PPO", lambda: [finetuned_models["PPO"].sample_path(tm.tasks["TASK1"].start_state) for _ in range(K)]))
        
        # Add Reinforce-rej
        if "Reinforce-rej" in finetuned_models:
            strategies.append(("Reinforce-rej", lambda: [finetuned_models["Reinforce-rej"].sample_path(tm.tasks["TASK1"].start_state) for _ in range(K)]))
    
    # Add PPO-Sampling (GRPO-KL)
    if ppo_sampler:
        strategies.append(("GRPO-KL", lambda: [ppo_sampler.sample_path(temperature=0.5, task_id="TASK1") for _ in range(K)]))
    
    # Add inference methods
    strategies.extend([
        ("ORM-TAS", lambda: sampler.temperature_adjusted_sampling(num_samples=K, temperature=0.5, N=N)),
        ("ORM-BoN", lambda: sampler.outcome_bon_sampling(num_samples=K, N=N)),
        ("PRM-BoN", lambda: [sampler.potential_bon_sampling(partial_path=[tm.tasks["TASK1"].start_state], N=N) for _ in range(K)]),
        ("DPRM-BoN", lambda: [sampler.dprm_bon_sampling(partial_path=[tm.tasks["TASK1"].start_state], N=N, temperature=0.5) for _ in range(K)]),
        ("DPRM", lambda: [sampler.dprm_sampling(partial_path=[tm.tasks["TASK1"].start_state], temperature=0.5, N=N) for _ in range(K)]),
    ])
    
    # Create task-specific strategies for TASK2
    strategies_task2 = [
        ("Base Model", lambda: [tm.sample_path() for _ in range(K)]),
    ]
    
    # Add Finetune methods for TASK2 (in correct order)
    if finetuned_models:
        # Add REINFORCE
        if "REINFORCE" in finetuned_models:
            strategies_task2.append(("REINFORCE", lambda: [finetuned_models["REINFORCE"].sample_path(tm.tasks["TASK2"].start_state) for _ in range(K)]))
        
        # Add RAFT
        if "RAFT" in finetuned_models:
            strategies_task2.append(("RAFT", lambda: [finetuned_models["RAFT"].sample_path(tm.tasks["TASK2"].start_state) for _ in range(K)]))
        
        # Add PPO
        if "PPO" in finetuned_models:
            strategies_task2.append(("PPO", lambda: [finetuned_models["PPO"].sample_path(tm.tasks["TASK2"].start_state) for _ in range(K)]))
        
        # Add Reinforce-rej
        if "Reinforce-rej" in finetuned_models:
            strategies_task2.append(("Reinforce-rej", lambda: [finetuned_models["Reinforce-rej"].sample_path(tm.tasks["TASK2"].start_state) for _ in range(K)]))
    
    # Add PPO-Sampling (GRPO-KL) for TASK2
    if ppo_sampler:
        strategies_task2.append(("GRPO-KL", lambda: [ppo_sampler.sample_path(temperature=0.5, task_id="TASK2") for _ in range(K)]))
    
    # Add inference methods for TASK2
    strategies_task2.extend([
        ("ORM-TAS", lambda: sampler.temperature_adjusted_sampling(num_samples=K, temperature=0.5, N=N)),
        ("ORM-BoN", lambda: sampler.outcome_bon_sampling(num_samples=K, N=N)),
        ("PRM-BoN", lambda: [sampler.potential_bon_sampling(partial_path=[tm.tasks["TASK2"].start_state], N=N) for _ in range(K)]),
        ("DPRM-BoN", lambda: [sampler.dprm_bon_sampling(partial_path=[tm.tasks["TASK2"].start_state], N=N, temperature=0.5) for _ in range(K)]),
        ("DPRM", lambda: [sampler.dprm_sampling(partial_path=[tm.tasks["TASK2"].start_state], temperature=0.5, N=N) for _ in range(K)]),
    ])
    
    # Store results for both tasks
    task1_performance = []
    task2_performance = []
    task1_coverage = {"easy": [], "hard": [], "invalid": []}
    task2_coverage = {"easy": [], "hard": [], "invalid": []}
    strategy_names = []
    
    # Evaluate each strategy for TASK1
    for name, sample_func in strategies:
        print(f"Evaluating strategy for TASK1: {name}")
        strategy_names.append(name)
        
        # Task 1 Performance
        total_success_task1 = 0
        for question_idx, correct_cots in enumerate(task1_instances):
            question_success = False
            paths = sample_func()
            for path in paths:
                if any(tuple(path) == tuple(cot) for cot in correct_cots):
                    question_success = True
                    break
            if question_success:
                total_success_task1 += 1
        task1_performance.append(total_success_task1 / num_trials)
        
        # Task 1 Coverage
        easy_count_task1 = hard_count_task1 = invalid_count_task1 = 0
        total_samples_task1 = 0
        for _ in range(num_trials):
            paths = sample_func()
            for path in paths:
                total_samples_task1 += 1
                if tm.is_valid_path_for_task(path, "TASK1"):
                    if tm.is_easy_path(path):
                        easy_count_task1 += 1
                    else:
                        hard_count_task1 += 1
                else:
                    invalid_count_task1 += 1
        
        task1_coverage["easy"].append(easy_count_task1 / total_samples_task1)
        task1_coverage["hard"].append(hard_count_task1 / total_samples_task1)
        task1_coverage["invalid"].append(invalid_count_task1 / total_samples_task1)
    
    # Evaluate each strategy for TASK2
    task2_performance = []
    task2_coverage = {"easy": [], "hard": [], "invalid": []}
    
    for name, sample_func in strategies_task2:
        print(f"Evaluating strategy for TASK2: {name}")
        
        # Task 2 Performance
        total_success_task2 = 0
        for question_idx, correct_cots in enumerate(task2_instances):
            question_success = False
            paths = sample_func()
            for path in paths:
                if any(tuple(path) == tuple(cot) for cot in correct_cots):
                    question_success = True
                    break
            if question_success:
                total_success_task2 += 1
        task2_performance.append(total_success_task2 / num_trials)
        
        # Task 2 Coverage
        easy_count_task2 = hard_count_task2 = invalid_count_task2 = 0
        total_samples_task2 = 0
        for _ in range(num_trials):
            paths = sample_func()
            for path in paths:
                total_samples_task2 += 1
                if tm.is_valid_path_for_task(path, "TASK2"):
                    if tm.is_easy_path(path):
                        easy_count_task2 += 1
                    else:
                        hard_count_task2 += 1
                else:
                    invalid_count_task2 += 1
        
        task2_coverage["easy"].append(easy_count_task2 / total_samples_task2)
        task2_coverage["hard"].append(hard_count_task2 / total_samples_task2)
        task2_coverage["invalid"].append(invalid_count_task2 / total_samples_task2)
    
    # Print combined performance table
    print(f"\nPass@{K} Performance for Both Tasks:")
    print("| Strategy | TASK1 Pass@K Rate (%) | TASK2 Pass@K Rate (%) |")
    print("|----------|----------------------|----------------------|")
    for i, name in enumerate(strategy_names):
        task1_pct = task1_performance[i] * 100
        task2_pct = task2_performance[i] * 100
        print(f"| {name} | {task1_pct:.2f}% | {task2_pct:.2f}% |")
    
    # Print combined coverage table
    print(f"\nValid CoT Coverage for Both Tasks:")
    print("| Strategy | TASK1 Valid Easy CoTs (%) | TASK1 Valid Hard CoTs (%) | TASK1 Invalid CoTs (%) | TASK2 Valid Easy CoTs (%) | TASK2 Valid Hard CoTs (%) | TASK2 Invalid CoTs (%) |")
    print("|----------|---------------------------|---------------------------|------------------------|---------------------------|---------------------------|------------------------|")
    for i, name in enumerate(strategy_names):
        task1_easy_pct = task1_coverage["easy"][i] * 100
        task1_hard_pct = task1_coverage["hard"][i] * 100
        task1_invalid_pct = task1_coverage["invalid"][i] * 100
        task2_easy_pct = task2_coverage["easy"][i] * 100
        task2_hard_pct = task2_coverage["hard"][i] * 100
        task2_invalid_pct = task2_coverage["invalid"][i] * 100
        print(f"| {name} | {task1_easy_pct:.2f}% | {task1_hard_pct:.2f}% | {task1_invalid_pct:.2f}% | {task2_easy_pct:.2f}% | {task2_hard_pct:.2f}% | {task2_invalid_pct:.2f}% |")
    
    # Plot Task 1 Performance
    plt.figure(figsize=(14, 8))
    plt.bar(strategy_names, task1_performance, color=tm.tasks["TASK1"].color)
    plt.title(f"Pass@{K} Performance for TASK1")
    plt.xlabel("Sampling Strategy")
    plt.ylabel(f"Pass@{K} Rate")
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / 'TASK1_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"TASK1 performance plot saved as {output_path}")
    
    # Plot Task 2 Performance
    plt.figure(figsize=(14, 8))
    plt.bar(strategy_names, task2_performance, color=tm.tasks["TASK2"].color)
    plt.title(f"Pass@{K} Performance for TASK2")
    plt.xlabel("Sampling Strategy")
    plt.ylabel(f"Pass@{K} Rate")
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / 'TASK2_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"TASK2 performance plot saved as {output_path}")
    
    # Plot Task 1 Coverage
    plt.figure(figsize=(16, 10))
    p1 = plt.bar(strategy_names, task1_coverage["invalid"], color='gray')
    p2 = plt.bar(strategy_names, task1_coverage["hard"], bottom=task1_coverage["invalid"], color='red')
    p3 = plt.bar(strategy_names, task1_coverage["easy"], 
                bottom=[i+h for i, h in zip(task1_coverage["invalid"], task1_coverage["hard"])], 
                color='green')
    plt.title(f"TASK1 Valid CoT Coverage (K={K}, N={N}, {num_trials} trials)")
    plt.xlabel("Sampling Strategy")
    plt.ylabel("Proportion of Sampled Paths")
    plt.legend((p1[0], p2[0], p3[0]), ('Invalid', 'Hard Valid', 'Easy Valid'))
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / 'TASK1_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"TASK1 coverage plot saved as {output_path}")
    
    # Plot Task 2 Coverage
    plt.figure(figsize=(16, 10))
    p1 = plt.bar(strategy_names, task2_coverage["invalid"], color='gray')
    p2 = plt.bar(strategy_names, task2_coverage["hard"], bottom=task2_coverage["invalid"], color='red')
    p3 = plt.bar(strategy_names, task2_coverage["easy"], 
                bottom=[i+h for i, h in zip(task2_coverage["invalid"], task2_coverage["hard"])], 
                color='green')
    plt.title(f"TASK2 Valid CoT Coverage (K={K}, N={N}, {num_trials} trials)")
    plt.xlabel("Sampling Strategy")
    plt.ylabel("Proportion of Sampled Paths")
    plt.legend((p1[0], p2[0], p3[0]), ('Invalid', 'Hard Valid', 'Easy Valid'))
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / 'TASK2_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"TASK2 coverage plot saved as {output_path}")

if __name__ == "__main__":
    args = parse_args()
    OUTPUT_DIR = args.output_dir

    # Unified parameter configuration
    PARAMS = {
        "L": 4,              
        "K": 30,             # Increase K to ensure sampling low-probability CoTs
        "N": 15,              
        "temperature": 0.5,  
        "mc_budget": 1000,
        "T1": 2000,
        "T2": 500,
        "eta": 0.05,
        "eval_interval": 100,
        "M0": 2,              # Increased to support TASK2
        "M": 2,              
        "C_size": 1,         
        "seed": 42,
        "num_trials": 200,
        # Finetune parameters
        "finetune_iterations": 1000,
        "finetune_lr": 0.05,
        "finetune_beta": 1.0,
        "epsilon_clip": 0.2,
        "finetune_methods": ["REINFORCE", "RAFT", "PPO", "Reinforce-rej"],
        "finetune_eval_interval": 100,
        "rej_threshold": 5,   # Threshold for Reinforce-rej method
    }

    if args.quick:
        PARAMS.update({
            "K": 10,
            "N": 5,
            "mc_budget": 100,
            "T1": 200,
            "T2": 50,
            "eval_interval": 50,
            "num_trials": 20,
            "finetune_iterations": 50,
            "finetune_eval_interval": 10,
        })
    if args.seed is not None:
        PARAMS["seed"] = args.seed
    if args.num_trials is not None:
        PARAMS["num_trials"] = args.num_trials
    if args.finetune_iterations is not None:
        PARAMS["finetune_iterations"] = args.finetune_iterations
    if args.mc_budget is not None:
        PARAMS["mc_budget"] = args.mc_budget

    np.random.seed(PARAMS["seed"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create TMC with support for TASK1 and TASK2
    tm = TMC(
        L=PARAMS["L"],
        M0=PARAMS["M0"], 
        M=PARAMS["M"], 
        C_size=PARAMS["C_size"], 
        seed=PARAMS["seed"]
    )

    # Print TMC structure information
    tm.print_tmc_info()

    # Train linear model (pretrain TMC)
    state_sizes = [len(tm.S[i]) for i in range(tm.L)]
    linear_model = LinearSoftmaxModel(state_sizes)
    linear_model.train(
        tm,
        T1=PARAMS["T1"],
        T2=PARAMS["T2"],
        eta=PARAMS["eta"],
        eval_interval=PARAMS["eval_interval"]
    )

    # Create reward model (for TASK1)
    reward_model = RewardModel(tm, mc_budget=PARAMS["mc_budget"])
    
    # Generate Question Instances for both tasks
    test_question_instances_task1 = generate_question_instances(tm, PARAMS["num_trials"], "TASK1")
    test_question_instances_task2 = generate_question_instances(tm, PARAMS["num_trials"], "TASK2")
    
    # Generate separate Question Instances for finetuning (Task 1 only for finetuning)
    finetune_question_instances = generate_question_instances(tm, PARAMS["num_trials"], "TASK1")
    
    # ==================== Finetune Section =====================
    finetuned_models = {}
    
    # Apply various finetune methods
    for method in PARAMS["finetune_methods"]:
        # Create copy of pretrained model for finetuning
        finetune_model = LinearSoftmaxModel(state_sizes)
        finetune_model.thetas = {l: np.copy(linear_model.thetas[l]) for l in linear_model.thetas}
        
        # Finetune model
        finetune_model.finetune(
            tm, 
            reward_model,
            finetune_question_instances,
            method=method,
            iterations=PARAMS["finetune_iterations"],
            lr=PARAMS["finetune_lr"],
            beta=PARAMS["finetune_beta"],
            epsilon_clip=PARAMS["epsilon_clip"],
            N=PARAMS["N"],
            rej_threshold=PARAMS["rej_threshold"],
            eval_k=PARAMS["K"],
            progress_interval=PARAMS["finetune_eval_interval"]
        )
        
        # Save finetuned model
        finetuned_models[method] = finetune_model
        finetune_model.plot_finetune_progress()
    
    # Create PPO-Sampler
    ppo_sampler = PPOSampler(tm, reward_model, linear_model)
    
    # ==================== Evaluation =====================
    # Evaluate both tasks using the unified function
    evaluate_all_tasks(tm, reward_model, test_question_instances_task1, test_question_instances_task2,
                       linear_model=finetuned_models["REINFORCE"],
                       finetuned_models=finetuned_models,
                       ppo_sampler=ppo_sampler,
                       K=PARAMS["K"], N=PARAMS["N"], num_trials=PARAMS["num_trials"],
                       show_plot=False)
