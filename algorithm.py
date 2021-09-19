# Calculate mixing weights
import numpy as np
import scipy
from scipy import linalg

def get_weights(E_bs_hats, bs_star, var_squared):
    C = E_bs_hats.dot(E_bs_hats.T)
    K = bs_star.dot(E_bs_hats.T)
    return linalg.solve(C + np.diag(var_squared), K.T, assume_a='sym').T

def get_ols_weights(bstar_ests, d, sigmas):
    return get_weights(bstar_ests, bstar_ests, d*(sigmas**2))

def get_pca_weights(Sigma_stars, var_squared):
    C = Sigma_stars.dot(Sigma_stars.T)
    return linalg.solve(C + np.diag(var_squared), C, assume_a='sym').T, C

def update_cov(W, S_is):  
  return W.dot(S_is)