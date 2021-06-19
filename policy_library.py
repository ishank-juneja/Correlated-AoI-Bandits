import numpy as np
from math import log, sqrt
from numpy.random import beta, binomial, randint, normal


# Function to get the set of reference arms S_t
# to be used for the C-Bandit algorithms CUCB and CTS
def get_S_t(nsamps, t):
    # List of reference arms S_t
    S_t = []
    # Get number of arms
    K = len(nsamps)
    # Calculate threshold that qualifies as often enough
    thresh = (t-1)/K
    # Get the collection of reference arms S_t
    for i in range(K):
        if nsamps[i] >= thresh:
            S_t.append(i)
    # Return the boolean array identifying the reference arms
    return S_t


# Get the arm k_emp(t), the arm with highest empirical
# mean among the set of reference arms
def get_k_emp(p_estimates, S_t):
    # Array of all zeros
    S_t_boolean = np.zeros_like(p_estimates)
    S_t_boolean[S_t] = 1
    # Dot product between means and boolean
    # array identifying the reference arms
    reference_means = S_t_boolean * p_estimates
    k_emp = np.argmax(reference_means)
    return k_emp


# Identify the set of empirically competitive arms A_t
def get_A_t(phi_estimates, S_t, k_emp, mu_k_emp):
    # get min_{l in S_t} \hat{\phi}_{k, l} (t) if S_t is non-empty
    # Here we want the minimum in each column
    # S_t won't be empty since by construction at least
    # one arm will have more then (t-1)/K pulls
    phi_hat_min = np.min(phi_estimates[S_t], axis=0)
    is_comp = phi_hat_min >= mu_k_emp
    # Must add the arm with arm-index k_mp to this set
    is_comp[k_emp] = True
    return is_comp


# pestimates are empirical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB(p_estimates, nsamps, t):
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    k = np.argmax(I_ucb)
    return k


# pestimates are emperical estimate of probabilities
# nsamps is number of times each arm is sampled
def UCB_new(p_estimates, nsamps, AoI, t):
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / (nsamps + AoI))
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    k = np.argmax(I_ucb)
    return k


# Function to implement thompson sampling algorithm regret minimization
def thompson(s_arms, f_arms):
    n_arms = len(s_arms)
    # Array to hold observed samples
    samples = np.zeros_like(s_arms)
    for i in range(n_arms):
        # Create and sample a beta random variable for current arm
        samples[i] = beta(s_arms[i] + 1, f_arms[i] + 1)
    # Return the index of the largest sample
    k = np.argmax(samples)
    return k


# Function to implement thompson sampling algorithm regret minimization
def thompson_Normal(p_estimates, nsamps):
    # beta parameter for variance scaling
    beta = 1.0
    n_arms = len(p_estimates)
    # Array to hold observed samples
    samples = np.zeros_like(p_estimates)
    for i in range(n_arms):
        # Create and sample a beta random variable for current arm
        samples[i] = normal(p_estimates[i], sqrt(beta/(nsamps[i] + 1)))
    # Return the index of the largest sample
    k = np.argmax(samples)
    return k


# pestimates are emperical estimate of probabilities
# nsamps is number of times each arm is sampled
def QUCB(p_estimates, nsamps, t):
    n_arms = len(p_estimates)
    # Get exploration parameter by sampling
    # a bernoulli distribution
    E = binomial(1, min(1, 3 * n_arms * log(t) * log(t)/t))
    # Uniform random exploration
    if E == 1:
        k = randint(n_arms)
    else:
        # Update ucb value
        I_ucb = p_estimates + np.sqrt(log(t) * log(t) / (2*nsamps))
        # Determine arm to be sampled in current step,
        # Ties are broken by lower index preference
        k = np.argmax(I_ucb)
    return k


# Function to implement thompson sampling algorithm regret minimization
def Qthompson(s_arms, f_arms, t):
    n_arms = len(s_arms)
    # Array to hold observed samples
    samples = np.zeros_like(s_arms)
    # Get exploration parameter by sampling
    # a bernoulli distribution
    E = binomial(1, min(1, 3 * n_arms * log(t) * log(t) / t))
    # Uniform random exploration
    if E == 1:
        k = randint(n_arms)
    else:
        for i in range(n_arms):
            # Create and sample a beta random variable for current arm
            samples[i] = beta(s_arms[i] + 1, f_arms[i] + 1)
        # Return the index of the largest sample
        k = np.argmax(samples)
    return k


def CUCB(p_estimates, phi_estimates, nsamps, t):
    n_arms = len(p_estimates)
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Identify arms competitive wrt arm k_max using the
    # S_t, k_emp(t) and A_t C-Bandit formulation
    # Get set of reference arms S_t
    S_t = get_S_t(nsamps, t)
    # Get reference arm with highest mean k_emp
    k_emp = get_k_emp(p_estimates, S_t)
    mu_k_emp = p_estimates[k_emp]
    # Get boolean array for set of competitive arms A_t union with {k_emp}
    is_comp = get_A_t(phi_estimates, S_t, k_emp, mu_k_emp)
    max_index = 0
    k = 0
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    for i in range(n_arms):
        if I_ucb[i] > max_index and is_comp[i]:
            k = i  # Update arm
            max_index = I_ucb[i]
    return k


def CUCB_old(p_estimates, phi_estimates, nsamps, t):
    n_arms = len(p_estimates)
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Determine the arm that has been pulled the most number of times uptill
    # iteration t - 1
    k_max = np.argmax(nsamps)
    # Get pseudo gaps wrt arm k_max, second term on RHS is a vector
    del_hat = p_estimates[k_max] - phi_estimates[k_max]
    # Identify arms competitive wrt arm k_max
    is_comp = del_hat <= 0
    max_index = 0
    k = 0
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    for i in range(n_arms):
        if I_ucb[i] > max_index and is_comp[i]:
            k = i  # Update arm
            max_index = I_ucb[i]
    return k


def Cthompson(p_estimates, phi_estimates, nsamps, t, s_arms, f_arms):
    n_arms = len(p_estimates)
    samples = np.zeros_like(s_arms)
    # Identify arms competitive wrt arm k_max using the
    # S_t, k_emp(t) and A_t C-Bandit formulation
    # Get set of reference arms S_t
    S_t = get_S_t(nsamps, t)
    # Get reference arm with highest mean k_emp
    k_emp = get_k_emp(p_estimates, S_t)
    mu_k_emp = p_estimates[k_emp]
    # Get boolean array for set of competitive arms A_t union with {k_emp}
    is_comp = get_A_t(phi_estimates, S_t, k_emp, mu_k_emp)
    max_sample = 0
    k = 0
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    for i in range(n_arms):
        samples[i] = beta(s_arms[i] + 1, f_arms[i] + 1)
        if samples[i] > max_sample and is_comp[i]:
            k = i  # Update arm
            max_sample = samples[i]
    return k


# Thompson Sampling with normal distribution sample
def Cthompson_Normal(p_estimates, phi_estimates, nsamps, t):
    # Hyper-parameter that tunes the variance/standard deviation of samples
    # Strictly speaking we need \Beta > 1 but Beta = 1 works fine too
    beta = 1.0
    n_arms = len(p_estimates)
    samples = np.zeros_like(p_estimates)
    # Identify arms competitive wrt arm k_max using the
    # S_t, k_emp(t) and A_t C-Bandit formulation
    # Get set of reference arms S_t
    S_t = get_S_t(nsamps, t)
    # Get reference arm with highest mean k_emp
    k_emp = get_k_emp(p_estimates, S_t)
    mu_k_emp = p_estimates[k_emp]
    # Get boolean array for set of competitive arms A_t union with {k_emp}
    is_comp = get_A_t(phi_estimates, S_t, k_emp, mu_k_emp)
    max_sample = 0
    k = 0
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    for i in range(n_arms):
        # Uses standard deviation not variance to get normal samples
        samples[i] = normal(p_estimates[i], sqrt(beta/(nsamps[i] + 1)))
        if samples[i] > max_sample and is_comp[i]:
            k = i  # Update arm
            max_sample = samples[i]
    return k


# Old version of C-bandit algorithm with only 1 reference arm
def Cthompson_old(p_estimates, phi_estimates, s_arms, f_arms):
    n_arms = len(p_estimates)
    samples = np.zeros_like(s_arms)
    # Determine the arm that has been pulled the most number of times uptill
    # iteration t - 1
    k_max = np.argmax(s_arms + f_arms)
    # Get pseudo gaps wrt arm k_max, second term on RHS is a vector
    del_hat = p_estimates[k_max] - phi_estimates[k_max]
    # Identify arms competitive wrt arm k_max
    is_comp = del_hat <= 0
    max_sample = 0
    k = 0
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    for i in range(n_arms):
        samples[i] = beta(s_arms[i] + 1, f_arms[i] + 1)
        if samples[i] > max_sample and is_comp[i]:
            k = i  # Update arm
            max_sample = samples[i]
    return k


def U_CUCB(p_estimates, dist_hat, arm_list, eps, nsamps, t):
    n_arms = len(p_estimates)
    # Update ucb value
    I_ucb = p_estimates + np.sqrt(2 * log(t) / nsamps)
    # Get indices that would sort distribution array
    dist_sort = np.argsort(dist_hat)
    # Reverse for descending
    dist_sort = dist_sort[::-1]
    # Update eps confidence set
    Cstar = []
    sum_prob = 0
    ctr = 0
    while sum_prob < 1 - eps:
        sum_prob = sum_prob + dist_hat[dist_sort[ctr]]
        Cstar.append(dist_sort[ctr])
        ctr = ctr + 1
    # Determine arm to be sampled in current step,
    # Ties are broken by lower index preference
    # Init list for competitive arms, initially assume all are competitive
    comp_arms = list(range(n_arms))
    # Get current empirical estimates
    pseudo_exp = np.dot(dist_hat, np.transpose(arm_list))
    # determine the competitive set with respect to Cstar
    for i in range(n_arms):
        for j in range(n_arms):
            # If the ith arm func is less the j th arm func for all x in Cstar
            if np.all(arm_list[i][Cstar] < arm_list[j][Cstar]) and pseudo_exp[i] \
                    < pseudo_exp[j] and i in comp_arms:
                # Remove from competitive set
                comp_arms.remove(i)
    # Determine arm to be sampled in current step, Ties are broken by index preference
    max_ucb = 0
    k = 0
    for i in comp_arms:
        if I_ucb[i] > max_ucb:
            k = i  # Update arm
            max_ucb = I_ucb[i]
    return k


# Library only
if __name__ == '__main__':
    exit(0)
