import numpy as np
from scipy.stats import entropy
import random
from typing import List, Dict
import matplotlib.pyplot as plt

# RH: Helper routine which shows working KL Div maximization routine
# Effectively minimizes mutual information

# --- Configuration ---
LEARNING_RATE = 0.0001
NUM_ITERATIONS = 1000
NUM_STATES = 100

def calculate_kl_divergence(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculates the KL divergence D_KL(P1 || P2) using scipy.stats.entropy.

    D_KL(P1 || P2) = sum(P1(i) * log(P1(i) / P2(i)))
    
    Args:
        p1: The probability distribution P1 (the reference).
        p2: The probability distribution P2 (the target).
        
    Returns:
        The KL divergence value.
    """
    # Small epsilon value to handle zeros, required by the KL divergence definition
    epsilon = 1e-12 
    
    # Ensure P1 and P2 are valid distributions (sum to 1) and non-negative
    p1 = p1 / np.sum(p1)
    p2 = p2 / np.sum(p2)
    
    # Replace P1(i) and P2(i) with epsilon if they are 0 to handle the log domain
    p1_safe = np.where(p1 > epsilon, p1, epsilon)
    p2_safe = np.where(p2 > epsilon, p2, epsilon)
    
    # The scipy function entropy(pk, qk) calculates D_KL(pk || qk)
    # where pk is P1 and qk is P2.
    kl_div = entropy(p1_safe, qk=p2_safe)
    
    return kl_div

def gradient_ascent_update(p1_old: np.ndarray, p2: np.ndarray, learning_rate: float) -> np.ndarray:
    """
    Performs a gradient ascent step to maximize D_KL(P1 || P2).
    
    The gradient of D_KL(P1 || P2) with respect to P1 is:
    nabla_P1 D_KL = log(P1 / P2) + 1
    
    The update rule is: P1_new = P1_old + learning_rate * nabla_P1 D_KL
    
    Args:
        p1_old: The current P1 distribution (numpy array).
        p2: The fixed P2 distribution (numpy array).
        learning_rate: The step size for ascent.
        
    Returns:
        The new, normalized P1 distribution.
    """
    epsilon = 1e-12 
    p1_safe = np.where(p1_old > epsilon, p1_old, epsilon)
    p2_safe = np.where(p2 > epsilon, p2, epsilon)
    
    # 1. Calculate the gradient
    # This is the derivative *before* considering the P1 normalization constraint
    gradient = np.log(p1_safe / p2_safe) + 1.0
    
    # 2. Perform the gradient ascent step
    p1_new_unconstrained = p1_old + learning_rate * gradient
    
    # 3. Apply constraints (P1 >= 0 and sum(P1) = 1)
    # Ensure all probabilities are non-negative
    p1_new_unconstrained[p1_new_unconstrained < 0] = 0
    
    # Normalize the distribution to ensure sum(P1) = 1
    p1_new_normalized = p1_new_unconstrained / np.sum(p1_new_unconstrained)

    # RH: This allows for less sparse distributions, but also results in noisier gradient ascent solution
    # RH:   also should technically be applied before clipping operation is performed at 0 boundary
    # p1_new_normalized = (p1_new_unconstrained + np.random.normal(0, 1/(10*p1_new_unconstrained.size), p1_new_unconstrained.size)) / np.sum(p1_new_unconstrained)
    
    return p1_new_normalized

def maximize_kldiv(p1, p2, niter=100, lr=0.0001):
    # Run maximization loop
    for i in range(niter):
        p1 = gradient_ascent_update(p1, p2, learning_rate=lr)
    return p1

def run_maximization_experiment():
    """Runs the iterative KL divergence maximization process."""
    print(f"--- KL Divergence Maximization (D_KL(P1 || P2)) ---")
    print(f"Using scipy.stats.entropy for calculation.")
    print(f"Total States: {NUM_STATES}, Iterations: {NUM_ITERATIONS}, LR: {LEARNING_RATE}\n")

    # 1. Define the fixed target distribution P2
    P2 = (P2:=np.random.uniform(0,1,100)) / P2.sum()
    
    # 2. Define the initial distribution P1 (e.g., uniform)
    P1 = np.full(100, 1.0 / 100)
    
    print(f"Initial P1 (Uniform): {P1.round(4)}")
    print(f"Fixed P2 (Target):    {P2.round(4)}\n")

    # --- Iterative Maximization ---
    
    history = []
    
    for i in range(1, NUM_ITERATIONS + 1):
        # 1. Calculate current KL divergence
        kl_div = calculate_kl_divergence(P1, P2)
        history.append(kl_div)
        
        # 2. Perform gradient ascent step
        P1 = gradient_ascent_update(P1, P2, LEARNING_RATE)
        
        if i % int(NUM_ITERATIONS/10) == 0 or i == NUM_ITERATIONS:
            print(f"--- Iteration {i} ---")
            print(f"  KL Divergence: {kl_div:.6f}")
            # Do not print P1 distribution here as it's too large (100 states)

    plt.plot(history)
    plt.title(r"KL Divergence $\uparrow$")
    plt.ylabel(r"$KL Div(p_1(x) || p_2(x))$")
    plt.xlabel("Iteration")
    plt.show()

    plt.plot(range(100), P1, P2, alpha=0.4)
    plt.show()
            
    print("\n--- Final Analysis ---")
    print(f"Maximum divergence achieved: {history[-1]:.6f}")
    
    # Print a sample of the history data for visualization in an external tool
    print("\n--- KL Divergence History Sample ---")
    print(f"Initial 20 values: {history[:20]}")
    print(f"Final 20 values:   {history[-20:]}")
    
    print(f"\nObservation: P1 successfully shifted mass away from P2 to maximize divergence.")
    
if __name__ == "__main__":
    run_maximization_experiment()