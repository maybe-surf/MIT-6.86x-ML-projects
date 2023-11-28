"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    from scipy.stats import multivariate_normal
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # the normal and weight, denominator the above but for every distr in theta
    means = mixture.mu
    varss = mixture.var
    ps = mixture.p
    soft_counts_all = np.empty((X.shape[0], len(ps)))
    LL = 0.
    for i in range(X.shape[0]):
        prob_x = 0
        soft_counts = []
        for j in range(len(ps)):
            numerator = ps[j]*multivariate_normal.pdf(X[i], mean=means[j], cov=varss[j])
            prob_x += numerator
            soft_counts.append(numerator)
        soft_counts = np.array(soft_counts)/prob_x
        soft_counts_all[i] = soft_counts
        LL += np.log(prob_x)
    return (soft_counts_all, LL)




def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    model_given_data = np.sum(post, axis=0)
    Ps = model_given_data/X.shape[0]
    mean_num = np.matmul(np.transpose(post), X)
    Means = np.empty((post.shape[1], X.shape[1]))
    Vars = []
    for j in range(post.shape[1]):
        mean_j = np.transpose(mean_num[j])/model_given_data[j]
        Means[j, :] = mean_j
        diff = np.sum((X - mean_j)**2, axis=1)
        var_num = np.dot(post[:, j], diff)
        Vars.append(var_num/(X.shape[1]*model_given_data[j]))
    Vars = np.array(Vars)
    return GaussianMixture(Means, Vars, Ps)
        
    

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    oldLL = "old"
    soft_counts, newLL = estep(X, mixture)
    while(oldLL=="old" or (newLL-oldLL) > (10**(-6)*abs(newLL))):
        oldLL = newLL
        mixture = mstep(X, soft_counts)
        soft_counts, newLL = estep(X, mixture)
    return (mixture, soft_counts, newLL)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
