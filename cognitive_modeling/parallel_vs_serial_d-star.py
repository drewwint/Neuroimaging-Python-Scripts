##############################################################################################
# Brain response specific to two separate stimuli to model parallel versus serial processing 
# Dr. Drew E. Winters, PhD. 
##############################################################################################

# Before running this function the user needs to extract 
  # 1) z maps from an experiment with data on the brain
  # 2) probability priors - should be done with a hidden Markov model and a mixture of binomial distributions or a maximum likelihood
# thsi is modified from dn to relax dependence on the number of items in the z vector

import scipy.stats as stats
import numpy as np

def d_star(z, p):
    """
    Calculate the normalized expected deviation (D*) of brain response to stimuli.
    Here we adjust Dn by using the mean so results are no longer dependent on 
    the number of items in z.  
    D* quantitates the cognitive processes of the brain in serial or parallel.
    A small D* suggests a more parallel form of processing, and a large D*
    suggests a more serial form of processing.
    This is intended to be used on experiments with competing stimuli such as
    a dual-task experiment.
    D* is the limit of Dn as n approaches infinity.

    Parameters:
    - z: List of Z scores from an fMRI experiment z map
    - p: List of prior probabilities of attending stimulus 1

    Returns:
    - D_star: Measure of parallel or serial processing (normalized expected deviation)
    """
    n = len(z)
    d_n = []

    for zz, prob in zip(z, p):
        b_p = stats.binom.pmf(abs(zz), n, prob)
        q = 1 - b_p
        d_n.append(abs(zz - n/2) * b_p)

    dn_star = np.mean(d_n) / (n/2)

    return dn_star

# Example usage
z_values = [3, 2.8, 2.1, -3, -0.4]
probabilities = [0.8, 0.2, 0.3, 0.2, 0.9]

divergence_result = d_star(z_values, probabilities)
print("Normalized Expected Deviation (D*):", divergence_result)

