import numpy as np
from bandit_class import ArmFunctions
from policy_library import *
import sys
import argparse


# Command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("-i", action="store", dest="file")
parser.add_argument("-STEP", action="store", dest="STEP", type=int)
parser.add_argument("-horizon", action="store", dest="horizon", type=float)
parser.add_argument("-nruns", action="store", dest="nruns", type=int)
# Optional Aoi_Aware argument
parser.add_argument("--AoI_aware", action='store_true')
args = parser.parse_args()
file = args.file
# Policies to be simulated, 'new' is an AoI aware version of UCB
# algos = ['ucb', 'ts', 'qucb', 'qts', 'cucb', 'cts', 'cucb-old', 'cts-old', 'u-cucb', 'new']
# algos = ['ucb', 'ts', 'cucb', 'cts']
algos = ['ucb', 'ts', 'ts-normal', 'cucb', 'cts', 'cts-normal']
# If the AoI_aware flag is specified in the cmd line, then add
if args.AoI_aware:
    # placeholder for the aoi versions of these algos
    aoi_algos = []
    for algo in algos:
        aoi_algos.append("aoi-aware-" + algo)
    # Merge the 2 lists into single complete list of policies
    algos = algos + aoi_algos
# Horizon/ max number of iterations
horizon = int(args.horizon)
# Number of runs to average over
nruns = args.nruns
# epsilon parameter for algorithms that need them
# Default value = 0, unused
eps = 0.0
# Step interval for which data is recorded
STEP = args.STEP


if __name__ == '__main__':
    # Read in file containing MAB instance information (X and arm functions)
    file_in = open(file, 'r')
    # Read in probability distribution over the mass for above support through line 1 of file
    dist_str = file_in.readline()
    # Convert read in string of values to float array
    dist = [float(x) for x in dist_str.split(' ')]
    # Convert to array
    dist = np.array(dist)
    # Generate support points as np array
    support = np.arange(len(dist))
    # Read in lines corresponding to arm functions
    functions_str = file_in.readlines()
    # Count number of arms
    n_arms = len(functions_str)
    # Convert strings to lists
    arm_list = []
    for i in range(n_arms):
        # Read in functions as 0 or 1 integers, integers > 1 are not permitted
        arm_list.append([int(s) for s in functions_str[i].split(' ')])
    # Covert to numpy array for sliced indexing
    arm_list = np.array(arm_list)
    # Check if any element of the arm functions lies outside of 0-1
    if arm_list.any() > 1 or arm_list.any() < 0:
        sys.exit("bandit instance must contain only 0-1 rewards")
    # Compute expected reward associated with each arm
    expectations = np.dot(dist, np.transpose(arm_list))
    # Get maximum expected reward among all arms (optimal channel)
    mu_max = np.max(expectations)
    # Get arm index for the optimal arm
    k_opt = np.argmax(expectations)
    # Assemble a correlated bandit instance object
    bandit_instance = ArmFunctions(arm_list)
    for al in algos:
        for rs in range(nruns):
            # Set numpy random seed to make output deterministic for a given run
            np.random.seed(rs)
            # Get horizon number of samples of the the discrete latent random variable X
            X_realize = np.random.choice(support, horizon, p=dist)
            # Initialise cumulative reward and cumulative AoI
            AoI_cum, REW = 0, 0
            # Init empirical conventional regret and AoI regret
            AoI_REG, REG = 0, 0
            # Initialise cumulative AoI and AoI of optimal arm
            AoI, AoI_star = 1, 1
            # UCB: Vanilla Upper Confidence Bound Sampling algorithm
            if al == 'ucb' or al == 'aoi-aware-ucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Now begin UCB based decisions
                for t in range(1, horizon + 1):
                    # To initialise estimates from all arms
                    if t < n_arms + 1:
                        # sample the arm with (array) index (t - 1)
                        k = t - 1
                    else:
                        # If current AoI is too high, be greedy if running an AoI-aware policy
                        if AoI > 1 / (np.max(mu_hat) + np.finfo(float).eps) and al[:9] == 'aoi-aware':
                            k = np.random.choice(np.flatnonzero(np.isclose(mu_hat, np.max(mu_hat))))
                        else:
                            # Update ucb index value for all arms based on quantities from
                            # previous iteration and obtain arm index to sample
                            k = UCB(mu_hat, nsamps, t)
                    # Get 0/1 reward based on arm/channel choice
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t-1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            # New experimental policy that incorporates current AoI a(t) into exploration
            # No AoI aware version for this since it incorporates state in a different manner
            elif al == 'new':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Now begin UCB based decisions
                for t in range(1, horizon + 1):
                    # To initialise estimates from all arms
                    if t < n_arms + 1:
                        # sample the arm with index (t - 1)
                        k = t - 1
                    else:
                        # Update ucb index value for all arms based on quantities from
                        # previous iteration and obtain arm index to sample
                        k = UCB_new(mu_hat, nsamps, AoI, t)
                    # Get 0/1 reward based on arm/channel choice
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t-1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))

            # Thompson Sampling algorithm with Beta-Priors
            elif al == 'ts' or al == 'aoi-aware-ts':
                # Array to record how many times a particular arm gave r = 1 (Success)
                s_arms = np.zeros(n_arms)  # Each sampled once at start
                # Array to record how many times a particular arm gave r = 0 (Failure)
                f_arms = np.zeros(n_arms)  # Each sampled once at start
                # Begin Thompson sampling loop
                for t in range(1, horizon + 1):
                    # If current AoI is too high
                    if AoI > 1 / (np.max(s_arms/(s_arms + f_arms + np.finfo(float).eps)) +
                                  np.finfo(float).eps) and al[:9] == 'aoi-aware':
                        k = np.random.choice(np.flatnonzero(np.isclose(s_arms/(s_arms + f_arms + np.finfo(float).eps),
                                                                       np.max(s_arms/(s_arms + f_arms +
                                                                                      np.finfo(float).eps)))))
                    else:
                        # Determine arm to be sampled in current step
                        k = thompson(s_arms, f_arms)
                    # Get 0/1 reward
                    r = bandit_instance.sample(k, X_realize[t-1])
                    # Update failures and successes of channel using reward r
                    s_arms[k] += r
                    f_arms[k] += 1 - r
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            # Thompson Sampling algorithm with Gaussian/Normal priors
            elif al == 'ts-normal' or al == 'aoi-aware-ts-normal':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Begin Thompson sampling loop
                for t in range(1, horizon + 1):
                    if AoI > 1 / (np.max(mu_hat) + np.finfo(float).eps) and al[:9] == 'aoi-aware':
                        k = np.random.choice(np.flatnonzero(np.isclose(mu_hat, np.max(mu_hat))))
                    else:
                        # Update ucb index value for all arms based on quantities from
                        # previous iteration and obtain arm index to sample
                        k = thompson_Normal(mu_hat, nsamps)
                    # Get 0/1 reward based on arm/channel choice
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            # Queue UCB algorithm
            elif al == 'qucb' or al == 'aoi-aware-qucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Number of times a certain arm is sampled, each arm is sampled once at start
                nsamps = np.zeros(n_arms, dtype=np.int32)
                # Now begin UCB based decisions
                for t in range(1, horizon + 1):
                    # To initialise estimates from all arms
                    # To initialise estimates from all arms
                    if t < n_arms + 1:
                        # sample the arm with index (t - 1)
                        k = t - 1
                    else:
                        # If current AoI is too high
                        if AoI > 1 / (np.max(mu_hat) + np.finfo(float).eps) and al[:9] == 'aoi-aware':
                            k = np.random.choice(np.flatnonzero(np.isclose(mu_hat, np.max(mu_hat))))
                        else:
                            # Update ucb index value for all arms based on quantities from
                            # previous iteration and obtain arm index to sample
                            k = QUCB(mu_hat, nsamps, t)
                    # Get 0/1 reward based on arm/channel choice
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            # Queue Thompson Sampling algorithm
            elif al == 'qts' or al == 'aoi-aware-qts':
                # Array to record how many times a particular arm gave r = 1 (Success)
                s_arms = np.zeros(n_arms)  # Each sampled once at start
                # Array to record how many times a particular arm gave r = 0 (Failure)
                f_arms = np.zeros(n_arms)  # Each sampled once at start
                # Begin Thompson sampling loop
                for t in range(1, horizon + 1):
                    # If current AoI is too high
                    if AoI > 1 / (np.max(s_arms / (s_arms + f_arms + np.finfo(float).eps)) +
                                  np.finfo(float).eps) and al[:9] == 'aoi-aware':
                        k = np.random.choice(np.flatnonzero(np.isclose(s_arms / (s_arms + f_arms + np.finfo(float).eps),
                                                                       np.max(s_arms / (s_arms + f_arms +
                                                                                        np.finfo(float).eps)))))
                    else:
                        # Determine arm to be sampled in current step
                        k = Qthompson(s_arms, f_arms, t)
                    # Get 0/1 reward
                    r = bandit_instance.sample(k, X_realize[t-1])
                    # Update failures and successes of channel using reward r
                    s_arms[k] += r
                    f_arms[k] += 1 - r
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            # Correlated UCB algorithm
            elif al == 'cucb' or al == 'aoi-aware-cucb':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Square matrix Array to hold empirical pseudo rewards
                phi_hat = np.zeros((n_arms, n_arms))
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Now begin C - UCB based decisions
                for t in range(1, horizon + 1):
                    # To initialise estimates from all arms
                    if t < n_arms + 1:
                        # sample the arm with index (t - 1)
                        k = t - 1
                    else:
                        # If current AoI is too high
                        if AoI > 1 / (np.max(mu_hat) + np.finfo(float).eps) and al[:9] == 'aoi-aware':
                            k = np.random.choice(np.flatnonzero(np.isclose(mu_hat, np.max(mu_hat))))
                        else:
                            # Update ucb index value for all arms based on quantities from
                            # previous iteration and obtain arm index to sample
                            k = CUCB(mu_hat, phi_hat, nsamps, t)
                    # Get 0/1 reward based on arm/channel choice
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Compute and update pseudo rewards
                    phi_hat[k] = ((nsamps[k] - 1) * phi_hat[k] + bandit_instance.get_pseudo_rew(k, r)) / nsamps[k]
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW,
                                                                                              AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            # Correlated UCB algorithm
            elif al == 'cucb-old' or al == 'aoi-aware-cucb-old':
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Square matrix Array to hold empirical pseudo rewards
                phi_hat = np.zeros((n_arms, n_arms))
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Now begin C - UCB based decisions
                for t in range(1, horizon + 1):
                    # To initialise estimates from all arms
                    # To initialise estimates from all arms
                    if t < n_arms + 1:
                        # sample the arm with index (t - 1)
                        k = t - 1
                    else:
                        # If current AoI is too high
                        if AoI > 1 / (np.max(mu_hat) + np.finfo(float).eps) and al[:9] == 'aoi-aware':
                            k = np.random.choice(np.flatnonzero(np.isclose(mu_hat, np.max(mu_hat))))
                        else:
                            # Update ucb index value for all arms based on quantities from
                            # previous iteration and obtain arm index to sample
                            k = CUCB_old(mu_hat, phi_hat, nsamps, t)
                    # Get 0/1 reward based on arm/channel choice
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Compute and update pseudo rewards
                    phi_hat[k] = ((nsamps[k] - 1) * phi_hat[k] + bandit_instance.get_pseudo_rew(k, r)) / nsamps[
                        k]
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW,
                                                                                              AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            elif al == 'cts' or al == 'aoi-aware-cts':
                # Array to record how many times a particular arm gave r = 1 (Success)
                s_arms = np.zeros(n_arms)  # Each sampled once at start
                # Array to record how many times a particular arm gave r = 0 (Failure)
                f_arms = np.zeros(n_arms)  # Each sampled once at start
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Square matrix Array to hold empirical pseudo rewards
                phi_hat = np.zeros((n_arms, n_arms))
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Begin Thompson sampling loop
                for t in range(1, horizon + 1):
                    # If current AoI is too high
                    if AoI > 1 / (np.max(s_arms / (s_arms + f_arms + np.finfo(float).eps)) +
                                  np.finfo(float).eps) and al[:9] == 'aoi-aware':
                        k = np.random.choice(np.flatnonzero(np.isclose(s_arms / (s_arms + f_arms + np.finfo(float).eps),
                                                                       np.max(s_arms / (s_arms + f_arms +
                                                                                        np.finfo(float).eps)))))
                    else:
                        # Determine arm to be sampled in current step
                        k = Cthompson(mu_hat, phi_hat, nsamps, t, s_arms, f_arms)
                    # Get 0/1 reward
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update failures and successes of channel using reward r
                    s_arms[k] += r
                    f_arms[k] += 1 - r
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Compute and update pseudo rewards
                    phi_hat[k] = ((nsamps[k] - 1) * phi_hat[k] + bandit_instance.get_pseudo_rew(k, r)) / nsamps[k]
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            elif al == 'cts-normal' or al == 'aoi-aware-cts-normal':
                # Array to record how many times a particular arm gave r = 1 (Success)
                s_arms = np.zeros(n_arms)  # Each sampled once at start
                # Array to record how many times a particular arm gave r = 0 (Failure)
                f_arms = np.zeros(n_arms)  # Each sampled once at start
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Square matrix Array to hold empirical pseudo rewards
                phi_hat = np.zeros((n_arms, n_arms))
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Begin Thompson sampling loop
                for t in range(1, horizon + 1):
                    # If current AoI is too high
                    if AoI > 1 / (np.max(s_arms / (s_arms + f_arms + np.finfo(float).eps)) +
                                  np.finfo(float).eps) and al[:9] == 'aoi-aware':
                        k = np.random.choice(np.flatnonzero(np.isclose(s_arms / (s_arms + f_arms + np.finfo(float).eps),
                                                                       np.max(s_arms / (s_arms + f_arms +
                                                                                        np.finfo(float).eps)))))
                    else:
                        # Determine arm to be sampled in current step
                        k = Cthompson_Normal(mu_hat, phi_hat, nsamps, t)
                    # Get 0/1 reward
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update failures and successes of channel using reward r
                    s_arms[k] += r
                    f_arms[k] += 1 - r
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Compute and update pseudo rewards
                    phi_hat[k] = ((nsamps[k] - 1) * phi_hat[k] + bandit_instance.get_pseudo_rew(k, r)) / nsamps[k]
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            elif al == 'cts-old' or al == 'aoi-aware-cts-old':
                # Array to record how many times a particular arm gave r = 1 (Success)
                s_arms = np.zeros(n_arms)  # Each sampled once at start
                # Array to record how many times a particular arm gave r = 0 (Failure)
                f_arms = np.zeros(n_arms)  # Each sampled once at start
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Square matrix Array to hold empirical pseudo rewards
                phi_hat = np.zeros((n_arms, n_arms))
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Begin Thompson sampling loop
                for t in range(1, horizon + 1):
                    # If current AoI is too high
                    if AoI > 1 / (np.max(s_arms / (s_arms + f_arms + np.finfo(float).eps)) +
                                  np.finfo(float).eps) and al[:9] == 'aoi-aware':
                        k = np.random.choice(np.flatnonzero(np.isclose(s_arms / (s_arms + f_arms + np.finfo(float).eps),
                                                                       np.max(s_arms / (s_arms + f_arms +
                                                                                        np.finfo(float).eps)))))
                    else:
                        # Determine arm to be sampled in current step
                        k = Cthompson_old(mu_hat, phi_hat, s_arms, f_arms)
                    # Get 0/1 reward
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update failures and successes of channel using reward r
                    s_arms[k] += r
                    f_arms[k] += 1 - r
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    # Compute and update pseudo rewards
                    phi_hat[k] = ((nsamps[k] - 1) * phi_hat[k] + bandit_instance.get_pseudo_rew(k, r)) / nsamps[k]
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            elif al == 'u-cucb' or al == 'aoi-aware-u-cucb':
                eps = 0.11
                # Array to hold estimated pseudo distribution of X, uniform prior
                dist_hat = np.repeat(1 / len(support), len(support))
                # dist_hat = np.repeat(TOL, len(support))
                # List to hold confidence set, initialise with complete support
                Cstar = list(range(n_arms))
                # Array to hold empirical estimates of each arms reward expectation
                mu_hat = np.zeros(n_arms)
                # Initialise UCB index with infinity so that each arm sampled once at start
                I_ucb = np.repeat(np.inf, n_arms)
                # Number of times a certain arm is sampled
                nsamps = np.zeros(n_arms)
                # Now begin C - UCB based decisions
                for t in range(1, horizon + 1):
                    # To initialise estimates from all arms
                    # To initialise estimates from all arms
                    if t < n_arms + 1:
                        # sample the arm with index (t - 1)
                        k = t - 1
                    else:
                        # If current AoI is too high
                        if AoI > 1 / (np.max(mu_hat) + np.finfo(float).eps) and al[:9] == 'aoi-aware':
                            k = np.random.choice(np.flatnonzero(np.isclose(mu_hat, np.max(mu_hat))))
                        else:
                            # Update ucb index value for all arms based on quantities from
                            # previous iteration and obtain arm index to sample
                            k = U_CUCB(mu_hat, dist_hat, arm_list, eps, nsamps, t)
                    # Get 0/1 reward based on arm/channel choice
                    r = bandit_instance.sample(k, X_realize[t - 1])
                    # Update conventional MAB reward
                    REW = REW + r
                    # Update current AoI
                    if r == 0:
                        # Failed transmission update AoI
                        AoI += 1
                    else:
                        # Successful transmission reset AoI
                        AoI = 1
                    # Increment cumulative AoI
                    AoI_cum += AoI
                    # What would have happened to AoI
                    # had we chosen optimal channel
                    r_star = arm_list[k_opt][X_realize[t - 1]]
                    if r_star == 0:
                        # Failed transmission update AoI
                        AoI_star += 1
                    else:
                        # Successful transmission reset AoI
                        AoI_star = 1
                    # Increment number of times kth arm sampled
                    nsamps[k] = nsamps[k] + 1
                    # Update empirical reward estimates, compute new empirical mean
                    mu_hat[k] = ((nsamps[k] - 1) * mu_hat[k] + r) / nsamps[k]
                    incr_indices = bandit_instance.get_inverse(k, r).astype(int)
                    # Update the pseudo distribution as described in Juneja et al.
                    dist_hat = (t * dist_hat + incr_indices / np.sum(incr_indices)) / (t + 1)
                    # Update empirical cumulative regret
                    REG += r_star - r
                    # Update empirical AoI regret
                    AoI_REG += AoI - AoI_star
                    REG = REG + arm_list[k_opt][X_realize[t - 1]] - r
                    # Record data at intervals of STEP in file
                    if t % STEP == 0:
                        sys.stdout.write(
                            "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al,
                                                                                              rs, eps, t, REG,
                                                                                              mu_max * t - REW, AoI_REG,
                                                                                              AoI_cum - t / mu_max))
            # invalid algorithm selected
            else:
                print("Invalid algorithm selected, ignored")
                continue
