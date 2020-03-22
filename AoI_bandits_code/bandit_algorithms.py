import numpy as np
from math import log
from numpy.random import beta


# pestimates are emperical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB(p_estimates, nsamps, t):
    # Update ucb value
    ucb = p_estimates + np.sqrt(2*log(t)/nsamps)
    # Ties are broken by index preference
    k = np.argmax(ucb)
    return k


# Function to implement thompson sampling algorithm regret minimization
def thompson(s_arms, f_arms):
    # Declare list of Beta random variables
    n_arms = len(s_arms)
    # Array to hold observed samples
    samples = np.zeros_like(s_arms)
    # Create a beta random variable for current arm
    # sample the variable and put it in samples array
    for i in range(n_arms):
        # Declare RV
        samples[i] = beta(s_arms[i] + 1, f_arms[i] + 1)
    # Return the index of the largest sample
    k = np.argmax(samples)
    return k


# Library only
if __name__ == '__main__':
    exit(0)
