import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import orth

# Generate data

def generate_central_bs_star(b_star, n, sigmas):
    d = b_star.shape[0]
    bs_star = []
    for ii in range(n):
        b = b_star + sigmas[ii]*np.random.randn(b_star.shape[0])
        bs_star.append(b)
    return np.vstack(bs_star)

def generate_orth_As(n, m, d):
    As = []
    for ii in range(n):
        A = np.random.randn(m, d)
        As.append(orth(A,0))
    return As

def generate_obs(A, b, sigma):
    return A.dot(b) + sigma*np.random.randn(A.shape[0])

def generate_n_obs(As, bs, sigmas):
    ys = []
    for A, b, sigma in zip(As, bs, sigmas):
        y = generate_obs(A, b, sigma)
        ys.append(y)
    return np.vstack(ys)

def generate_data_central_model(d, n, m, gt_sigma, obs_sigmas_mag):
    b_star = np.random.randn(d)
    bs_star = generate_central_bs_star(b_star, n, np.ones(n)*gt_sigma)
    As = generate_orth_As(n, m, d)
    sigmas = np.ones(n)*obs_sigmas_mag
    ys = generate_n_obs(As, bs_star, sigmas)
    
    return (bs_star, As, sigmas, ys)

def generate_data_star_player_model(d, n, m, gt_sigma, obs_sigmas_mag, mult_fac):
    b_star = np.random.randn(d)
    bs_star = generate_central_bs_star(b_star, n, np.ones(n)*gt_sigma)
    As = generate_orth_As(n, m, d)
    sigmas = np.ones(n)*obs_sigmas_mag*mult_fac
    sigmas[0] = obs_sigmas_mag
    ys = generate_n_obs(As, bs_star, sigmas)
    
    return (bs_star, As, sigmas, ys)

def generate_data_community_model(d, n, m, gt_sigma, obs_sigmas_mag, groups):
    bs_star, As, sigmas, ys = [], [], [], []

    for group in groups:
        group_n = int(n*group)
        bs_star_group, As_group, sigmas_group, ys_group = generate_data_central_model(d, group_n, m, gt_sigma, obs_sigmas_mag)

        bs_star.extend(bs_star_group)
        As.extend(As_group)
        sigmas.extend(sigmas_group)
        ys.extend(ys_group)

    bs_star, sigmas, ys = np.array(bs_star), np.array(sigmas), np.array(ys)
    return (bs_star, As, sigmas, ys)
        
# Calculate local estimates

def get_n_est(ys, As, f, params):
    bs_hats = []
    for y, A, param in zip(ys, As, params):
        b_hat = f(y, A, param)
        bs_hats.append(b_hat)
    return np.vstack(bs_hats)

def get_ridge(y, A, lambd_mult):
    return A.T.dot(y)*lambd_mult

def get_ols(y, A, dummy):
    return A.T.dot(y)


# Calculate mse

def get_mse(bs, bs_star):
    return np.sum((bs-bs_star)**2)/np.prod(bs.shape)

def get_input_snrs(sig, bs_star):
    d = bs_star.shape[1]
    return 10*np.log10(np.divide(np.sum(bs_star**2, 1), d*(sig**2)))

def get_output_snrs(bs, bs_star):
    return 10*np.log10(np.divide(np.sum(bs_star**2, 1), np.sum((bs-bs_star)**2, 1)))

def get_covariance_err(Sigma_hats, Sigma_stars):
    return np.mean((Sigma_hats - Sigma_stars)**2, axis=0)

def run_PCA(S, k):
    print(S.shape)
    w, v = np.linalg.eigh(S)
    U = v[:, -k:] 
    return U

def get_subspace_err(Sigma_hats, B_stars):
    res = []
    d = int(np.sqrt(Sigma_hats.shape[1]))
    k = B_stars[0].shape[1]
    
    for S, B in zip(Sigma_hats, B_stars):
        U = run_PCA(S.reshape(d,d), k) 
        err = np.sum((U.dot(U.T) - B.dot(B.T))**2)
        res.append(err/d)
    
    return res