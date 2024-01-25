##########################################################################
### Code to Calculate Dynamic Differential Covariance (DDC) in Python ####
##########################################################################
### Drew E. Winters, PhD. 
#
# Origional paper: https://doi.org/10.1073/pnas.2117234119
# Translated from Matlab code from: https://github.com/yschen13/DDC
# streamlined for python for faster code from previous translation
# from: https://github.com/drewwint/Neuroimaging-Python-Scripts/blob/main/Network_Analysis/dynam_diff_cov_ddc.py
#
#########################################################################

import numpy as np
from sklearn.linear_model import Ridge

def derivative_123_vectorized(f, dm, dt):
    t = np.arange(1 + dm, len(f) - dm)

    n_values = np.arange(1, dm + 1)
    n1, n2, n3 = np.meshgrid(n_values, n_values, n_values)

    D1 = np.sum(-((f[t - n2] * n1**3 - f[t + n2] * n1**3 - f[t - n1] * n2**3 + f[t + n1] * n2**3) /
                  (2 * dt * n1**3 * n2 - 2 * dt * n1 * n2**3)))

    D2 = np.sum(((f[t - n2] * n1**4 + f[t + n2] * n1**4 - f[t - n1] * n2**4 -
                   f[t + n1] * n2**4 - 2 * f[t] * (n1**4 - n2**4)) /
                  (dt**2 * n2**2 * (n1**4 - n1**2 * n2**2))))

    D3 = np.sum(((3 * (f[t - n3] * n1 * n2 * (n1**4 - n2**4) +
                       f[t + n3] * (-(n1**5 * n2) + n1 * n2**5) +
                       n3 * ((f[t - n1] - f[t + n1]) * n2 * (n2**4 - n3**4) +
                             f[t + n2] * (n1**5 - n1 * n3**4) +
                             f[t - n2] * (-n1**5 + n1 * n3**4)))) /
                  (dt**3 * n1 * (n1**2 - n2**2) * n3 * (n1**2 - n3**2) * (n2**3 - n2 * n3**2))))

    D1 /= dm**2
    D2 /= dm**2
    D3 /= dm**3

    return D1, D2, D3

def dCov_numerical_optimized(cx, h, dm=4):
    T, N = cx.shape
    
    # 1st order derivative computation
    diff_cx = np.diff(cx, axis=0) / h
    diff_cx = np.vstack([diff_cx, np.mean(diff_cx, axis=0)])
    Csample = np.cov(np.hstack([diff_cx, cx]), rowvar=False)
    dCov1 = Csample[:N, N:N + N]

    # 2nd order derivative computation
    diff_cx = (1/2 * cx[2:, :] - 1/2 * cx[:-2, :]) / h
    diff_cx = np.vstack([np.mean(diff_cx, axis=0), diff_cx, np.mean(diff_cx, axis=0)])
    Csample = np.cov(np.hstack([diff_cx, cx]), rowvar=False)
    dCov2 = Csample[:N, N:N + N]

    # Five-point stencil approximation
    diff_cx = np.convolve([-1, 8, -8, 1], cx, mode='valid') / (12 * h)
    diff_cx = np.vstack([np.mean(diff_cx, axis=0)] * 2 + [diff_cx] + [np.mean(diff_cx, axis=0)] * 2)
    Csample = np.cov(np.hstack([diff_cx, cx]), rowvar=False)
    dCov5 = Csample[:N, N:N + N]

    # Centered derivative from Taylor expansion
    diff_cx = np.array([derivative_123_vectorized(cx[:, i], dm, h)[0] for i in range(N)]).T
    cx_trunc = cx[1 + dm:T - dm, :]
    Csample = np.cov(np.hstack([diff_cx, cx_trunc]), rowvar=False)
    dCov_center = Csample[:N, N:N + N]

    return dCov1, dCov2, dCov5, dCov_center

def dCov_linear_Reg_optimized(V, TR, lamda):
    T, N = V.shape
    V_obs = (V - np.mean(V, axis=0)) / np.std(V, axis=0)
    dCov1, dCov2, _, dCov_center = dCov_numerical_optimized(V_obs, TR)
    Cov, _, B, _ = estimators(V_obs, 0, TR)
    C = dCov2
    A_reg = np.zeros_like(C)
    
    for i in range(C.shape[0]):
        ci = C[i, :]
        ridge_reg = Ridge(alpha=lamda)
        ridge_reg.fit(B, ci)
        A_reg[i, :] = ridge_reg.coef_

    return A_reg

