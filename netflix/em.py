"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    from scipy.stats import multivariate_normal
    means = mixture.mu
    varss = mixture.var
    ps = mixture.p
    soft_counts_all = np.empty((X.shape[0], len(ps)))
    LL = 0.
    for i in range(X.shape[0]):
        prob_x = 0
        soft_counts = []
        non_zero_inds = np.where(X[i] != 0)
        x_prime = X[i][non_zero_inds]
        for j in range(len(ps)):
            mu_prime = means[j][non_zero_inds]
            if(len(mu_prime)==0):
                numerator = ps[j]
                prob_x = 1
            else:
                numerator = ps[j]*multivariate_normal.pdf(x_prime, mean=mu_prime, cov=varss[j])
                prob_x += numerator
            soft_counts.append(numerator)
        soft_counts = np.array(soft_counts)/prob_x
        soft_counts_all[i] = soft_counts
        LL += np.log(prob_x)
    return (soft_counts_all, LL)



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    old_means = mixture.mu
    model_given_data = np.sum(post, axis=0)
    cu = np.empty(X.shape)
    cu = np.where(X == 0, 0, 1)
    mean_den = np.matmul(np.transpose(post), cu)
    #mgd_inc = np.sum(post_inc, axis=0)
    Ps = model_given_data/X.shape[0]
    mean_num = np.matmul(np.transpose(post), X)
    Means = np.empty((post.shape[1], X.shape[1]))
    Vars = []
    for j in range(post.shape[1]):
        mean_j = mean_num[j]/mean_den[j]
        Means[j, :] = mean_j

    for j in range(post.shape[1]):
        for l in range(X.shape[1]):
            mask = (X[:, l] != 0)  
            n_hat_l = np.sum(post[mask, j])  
            if n_hat_l < 1:
                Means[j, l] = old_means[j, l]
    for j in range(post.shape[1]):
        mean_j = Means[j]
        Means[j, :] = mean_j
        mean_j_ext = np.empty(X.shape)
        mean_j_ext = np.where(X==0, 0, 1)
        mean_j_ext = mean_j_ext*mean_j
        diff = np.sum((X - mean_j_ext)**2, axis=1)
        var_num = np.dot(post[:, j], diff)
        mag = np.sum(cu, axis=1)
        denom = np.dot(post[:, j], mag)
        Vars.append(var_num/denom)
    Vars = np.array(Vars)
    Vars = np.where(Vars < min_variance, min_variance, Vars)
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
    gmix = GaussianMixture(0, 0, 0)
    soft_counts, newLL = estep(X, mixture)
    while(oldLL=="old" or (newLL-oldLL) > (10**(-6)*abs(newLL))):
        oldLL = newLL
        mixture = mstep(X, soft_counts, gmix)
        soft_counts, newLL = estep(X, mixture)
    return (mixture, soft_counts, newLL)


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    from scipy.stats import multivariate_normal
    means = mixture.mu
    varss = mixture.var
    ps = mixture.p
    soft_counts_all = np.empty((X.shape[0], len(ps)))
    LL = 0.
    for i in range(X.shape[0]):
        prob_x = 0
        soft_counts = []
        non_zero_inds = np.where(X[i] != 0)
        x_prime = X[i][non_zero_inds]
        for j in range(len(ps)):
            mu_prime = means[j][non_zero_inds]
            if(len(mu_prime)==0):
                numerator = ps[j]
                prob_x = 1
            else:
                numerator = ps[j]*multivariate_normal.pdf(x_prime, mean=mu_prime, cov=varss[j])
                prob_x += numerator
            soft_counts.append(numerator)
        soft_counts = np.array(soft_counts)/prob_x
        soft_counts_all[i] = soft_counts
        LL += np.log(prob_x)
    X_pred = X.copy()
    to_predict = np.where(X_pred==0)
    X_pred[to_predict] = np.matmul(soft_counts_all, means)[to_predict]
    return X_pred
    
    


































