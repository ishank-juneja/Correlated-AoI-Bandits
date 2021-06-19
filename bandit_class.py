import numpy as np


# A class to define the functions of a structured multi-armed bandit instance
class ArmFunctions:
    # Parameterised Constructor with instance variable
    # Pass a list containing the function evaluations at certain X
    def __init__(self, arm_list):
        self.functions = arm_list
        self.narms = self.functions.shape[0]
        # Init array to hold pseudo rewards with respect to a single arm
        self.pseudo_rew = np.zeros(self.narms)

    # Function to implement sampling of the kth bandit arm at the t-th time step
    # Returns reward based on the realization of X at time t: x_t
    def sample(self, k, x_t):
        return self.functions[k][x_t]

    # Return psuedo reward of all arms with respect to arm k
    def get_pseudo_rew(self, k, rew):
        # Get set of possible x values/indices
        gk_inv_rew = np.where(self.functions[k] == rew)
        for i in range(self.narms):
            # Compute pseudo reward for arm i
            self.pseudo_rew[i] = np.max(np.take(self.functions[i], gk_inv_rew))
        return self.pseudo_rew

    # Return the value indices in the support corresponding to gk_inv(rew) as True and rest as false
    def get_inverse(self, k, rew):
        gk_inv_rew = self.functions[k] == rew
        # Return boolean array
        return gk_inv_rew
