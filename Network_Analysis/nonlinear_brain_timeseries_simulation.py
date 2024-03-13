
import numpy as np

def Nonlinear_simulation(G, deltaT, RandInput, options):
  ''' Define nonlinearity function based on options
      By default, simulate 1000 seconds
      dx/dt = G*F(x) + u 
      INPUT: 	
        G: connectivity pattern, G= [i,j]: connections from j to i
          Create a separate list for every node with a value representing its magnitude of connection. 
        deltaT: relevant for hte integration step: delta of time 
        RandInput: magnitude of noise in the simulation 
        options['nonlinearity']: specify either relu or sigmoid or sigmoid_sym
        options['parameter']: an parameter for relu offset or sigmoid slope that redresses bias
          - Could specify one but can vary or be tuned to your data.
          - See: https://www.jefkine.com/general/2016/08/24/formulating-the-relu/
      OUTPUT:
        V_pre: T (time points) x G(variables)

  '''
  if options['nonlinearity'] == 'relu':
      print('ReLu nonlinearity')
      F = relu
  elif options['nonlinearity'] == 'sigmoid':
      print('Sigmoid nonlinearity')
      F = sigmoid
  elif options['nonlinearity'] == 'sigmoid_sym':
      print('Symmetric Sigmoid nonlinearity')
      F = sigmoid_sym

  N = G.shape[0]
  T = int(options.get('Ttotal', 1000) / deltaT)  # Total simulation time
  V_pre = np.zeros((T, N))
  I = np.zeros((T, N))

  for t in range(1, T):
      u = np.random.randn(N) * RandInput
      I[t, :] = np.dot(G, F(V_pre[t - 1, :]))
      V_pre[t, :] = V_pre[t - 1, :] + (I[t, :] + u) * deltaT

      if np.any(V_pre[t, :] > 10000):
          print('Simulation exploded')
          break

      if t % (T // 10) == 0:
          print(f'Simulation iteration: {t / T:.1%}')

  return V_pre

def relu(x):
    return np.maximum(x - options['parameter'], 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-options['parameter'] * x))

def sigmoid_sym(x):
    return 1 / (1 + np.exp(-options['parameter'] * x)) - 0.5

# Example usage:
# G = np.array([[1, 0, 0], [0, 0.8, 0], [0, 0, 0.3]])  # Connectivity pattern matrix
# deltaT = 0.1
# RandInput = 0.1
# options = {'nonlinearity': 'relu', 'parameter': 1, 'Ttotal': 1000}

# V_pre = Nonlinear_simulation(G, deltaT, RandInput, options)
