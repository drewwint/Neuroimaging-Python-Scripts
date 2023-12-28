##############################################################################################
# Brain response specific to two separate stimuli to model parallel versus serial processing 
# Dr. Drew E. Winters, PhD. 
##############################################################################################

# Prior to running this function the user needs to extract 
  # 1) z maps from an experiment with data on the brain
  # 2) probability priors - should be done with a hidden Markov model and a mixture of binomial distributions or a maximum likelihood

import scipy.stats as stats
import numpy as np

def dn_bin(z, p):
    """
    Calculate the Deviation (Dn) of brain responses to stimuli in an fMRI experiment.
    
    Dn quantifies the cognitive processes of the brain, indicating whether the processing is more serial or parallel.
    A small Dn suggests a more parallel form of processing, while a large Dn suggests a more serial form of processing.
    This function is designed for experiments with competing stimuli, such as dual-task experiments.
    
    Dn can be interpreted as a normalized expected deviation between nodes attending to one stimulus and half of the total number of nodes.
    If the brain responses are split based on the attended stimulus, resulting in two proportions (summing to 1), Dn is the average difference between these two proportions.
    The Dn statistic can take values between 0 and 1.
    
    The Dn measure depends on the total number of regions (n) in the experiment.

    Parameters:
    - z (list): List of Z scores from an fMRI experiment z map.
    - p (list): List of prior probabilities of attending stimulus 1.

    Returns:
    - Dn (float): Measure of parallel or serial processing.
      - Lower values indicate parallel
      - Higher values indicate serial
    """
    # calculating n
    n = len(z)
    # Calculate the contribution of each data point to the Dn statistic
    d_n = []
    for zz, prob in zip(z, p):
        b_p = stats.binom.pmf(abs(zz), n, prob)
        q = 1 - b_p
        d_n.append(abs(zz - n / 2) * b_p)

    # Calculate the overall Dn statistic
    dn = np.sum(d_n) / (n / 2)

    return dn

# Example usage
z_values = [3, 2.8, 2.1, -3, -0.4]
probabilities = [0.8, 0.2, 0.3, 0.2, 0.9]

divergence_result = dn_bin(z_values, probabilities)
print("Divergence:", divergence_result)

# see: https://doi.org/10.1098/rsos.191553 for more details
