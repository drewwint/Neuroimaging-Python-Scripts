##########################################################################
### Code to Calculate Dynamic Differential Covariance (DDC) in Python ####
##########################################################################
### Drew E. Winters, PhD. 
#
# Origional paper: https://doi.org/10.1073/pnas.2117234119
#
#########################################################################

# Note: each function depends on eachother
  # After specifying all functions you can just run the dCov_linear_Reg()
  # This will get you a regularized DDC


import numpy as np
from sklearn.linear_model import Ridge


def derivative_123(f, dm, dt):
    t = np.arange(1 + dm, len(f) - dm)
    # For this function to work f has to be a NumPy array
    f = np.array(f)

    # D1 (First Derivative)
    D1 = 0
    d = 0
    for n1 in range(1, dm + 1):
        for n2 in range(n1 + 1, dm + 1):
            d += 1
            D1 += (-((f[t - n2] * n1**3 - f[t + n2] * n1**3 - f[t - n1] * n2**3 + f[t + n1] * n2**3) /
                     (2 * dt * n1**3 * n2 - 2 * dt * n1 * n2**3)))
    D1 /= d

    # D2 (Second Derivative)
    D2 = 0
    d = 0
    for n1 in range(1, dm + 1):
        for n2 in range(n1 + 1, dm + 1):
            d += 1
            D2 += ((f[t - n2] * n1**4 + f[t + n2] * n1**4 - f[t - n1] * n2**4 - f[t + n1] * n2**4 - 2 * f[t] * (n1**4 - n2**4)) /
                   (dt**2 * n2**2 * (n1**4 - n1**2 * n2**2)))
    D2 /= d

    # D3 (Third Derivative)
    D3 = 0
    d = 0
    for n1 in range(1, dm + 1):
        for n2 in range(n1 + 1, dm + 1):
            for n3 in range(n2 + 1, dm + 1):
                d += 1
                D3 += ((3 * (f[t - n3] * n1 * n2 * (n1**4 - n2**4) +
                             f[t + n3] * (-(n1**5 * n2) + n1 * n2**5) +
                             n3 * ((f[t - n1] - f[t + n1]) * n2 * (n2**4 - n3**4) +
                                   f[t + n2] * (n1**5 - n1 * n3**4) +
                                   f[t - n2] * (-n1**5 + n1 * n3**4)))) /
                       (dt**3 * n1 * (n1**2 - n2**2) * n3 * (n1**2 - n3**2) * (n2**3 - n2 * n3**2)))
    D3 /= d

    return D1, D2, D3



def dCov_numerical(cx, h, dm=4):
  """
  Linear dCov computation
  1st order derivative computation: dv/dt = (v(t+1)-v(t))/dt
  INPUT:
  	cx: T x N
  	h: sampling interval
        dm: averaging window: parameter related to dCov_center
  OUTPUT:
  	dCov1: 1st order
  	dCov2: 2nd order 
  	dCov5: five-point stencil approximation
        dCov_center: centered derivative from Taylor expansion
  NOTE: 
  	dCov = <dv/dt,v>
  	covariance is computed through cov()
  """
  T, N = cx.shape
  cx = np.array(cx)
  diff_cx = np.vstack(((cx[1:, :] - cx[:-1, :]) / h, np.mean(cx[1:, :] - cx[:-1, :], axis=0)))
  Csample = np.cov(np.hstack((diff_cx, cx)), rowvar=False)
  dCov1 = Csample[:N, N:N + N]

  diff_cx = np.vstack(((1/2 * cx[2:, :] - 1/2 * cx[:-2, :]) / h, np.mean(1/2 * cx[2:, :] - 1/2 * cx[:-2, :], axis=0),
                      diff_cx[-1, :]))
  Csample = np.cov(np.hstack((diff_cx, cx)), rowvar=False)
  dCov2 = Csample[:N, N:N + N]
  diff_cx = np.zeros((T, N))
  for i in range(2, T - 2):
      for j in range(2, N - 2):
        diff_cx[i, j] = (
            -cx[i + 2, j] + 8 * cx[i + 1, j]
            - 8 * cx[i - 1, j] + cx[i - 2, j]
        ) / 12.0
  Csample = np.cov(np.hstack((diff_cx, cx)), rowvar=False)
  dCov5 = Csample[:N, N:N + N]

  diff_cx = np.zeros(((T - 2 * dm)-1, N))
  for i in range(N):
      dx, _, _ = derivative_123(cx[:, i], dm, h)
      diff_cx[:, i] = dx

  cx_trunc = cx[1+dm:T - dm, :]
  Csample = np.cov(np.hstack((diff_cx, cx_trunc)), rowvar=False)
  dCov_center = Csample[:N, N:N + N]

  return dCov1, dCov2, dCov5, dCov_center


def estimators(V_obs, thres, TR):
  """
	INPUT:
  		V_obs: time points x variables
  		thres: Relu offset
  		TR: sampling interval (seconds)
  OUTPUT:
  		B: ReLu(x),x
  		dCov: dx/dt,x
  """
  T, N = V_obs.shape
  Cov = np.cov(V_obs, rowvar=False)
  precision = np.linalg.inv(Cov)
  Fx = np.maximum(V_obs - thres, 0)
  tmp = np.cov(np.hstack([Fx, V_obs]), rowvar=False)
  B = tmp[:N, N:]
  dV = np.diff(V_obs) / TR
  dV = np.vstack([np.mean(dV, axis=0), dV[1:-1], np.mean(dV, axis=0)])
  tmp = np.cov(np.hstack([dV, V_obs]), rowvar=False)
  dCov = tmp[:N, N:]

  return Cov, precision, B, dCov


def dCov_linear_Reg(V, TR, lamda):
  """
  L2 Regularized version of deltaL
  INPUT:
      V: time series
      TR: sampling interval
      lambda: regularization strength
  OUTPUT:
      A_reg; linear DDC with ridge regularization
  """
  T, N = V.shape
  V = np.array(V)
  V_obs = (V - np.mean(V, axis=0)) / np.std(V, axis=0)
  dCov1, dCov2, _, dCov_center = dCov_numerical(V_obs, TR)
  Cov, _, B, _ = estimators(V_obs, 0, TR)
  C = dCov2
  A_reg = np.zeros_like(C)
  
  for i in range(C.shape[0]):
      ci = C[i, :]
      ridge_reg = Ridge(alpha=lamda)
      ridge_reg.fit(B, ci)
      A_reg[i, :] = ridge_reg.coef_

  return A_reg
