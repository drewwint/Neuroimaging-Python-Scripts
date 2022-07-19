
# Communities and Modules

import matplotlib.pyplot as plt                 # plotting
import seaborn as sns                           # plotting
import pandas as pd                             # dataframes
import numpy as np                              # data manipulation
import networkx as nx                           # for network analysis
import bct                                      # brain connectivity toolbox
from sklearn.metrics import adjusted_rand_score # zrand score
from joblib import parallel_backend             # parallel processing


# to estimate communities, we run the louvain algorithm. 
  # We need to repeat the algorithm multiple times due to its non-deterministic nature.
num_reps  = 10
num_nodes = len(cij)

# estimating communities
  ##  run louvain algorithm num_reps times, retaining community estimates and
  ##  quality for each repetition.

ci = np.zeros([num_nodes,num_reps])
q = []
for i in range(0,num_reps):
  print(i)
  ci[:,i] = bct.community_louvain(np.array(cij))[0]
  q.append(bct.community_louvain(np.array(cij))[1])



#  are these communities statistically meaningful? to assess whether this is
  #  the case, we can compare their quality against the quality of partitions
  #  estimated from randomly rewired networks.

num_samples = 10
itera = 2**5


# generating random networks and estimate their communities and modularity 

q_rand = np.matrix(np.zeros([num_reps,num_samples])) # creating a matrix of 0's to insert values into

for i in range(0,num_samples):
  CIJ_rand = bct.randmio_und(np.array(cij),itera)[0]
  for ii in range(0,num_reps):
    print(ii)
    q_rand[ii,i] = bct.community_louvain(np.array(CIJ_rand))[1]


# calculate mean q for observed and randomized networks
mean_q      = np.mean(q)
mean_q_rand = np.mean(q_rand,axis=1)

# calculating p value
  # Calculate fraction of randomized networks with q greater than that of the observed network.  
p = np.mean(mean_q_rand >= mean_q)



a = bct.agreement(ci)/num_reps



# brain plots 

thr = 0.5
cicon = bct.consensus_und(a,thr,num_reps)
deg= bct.degrees_und(cij)

sz= scipy.interpolate.interp1d([min(deg),max(deg)],[1,100]) 
    # the first set specifies the length of values, 
    # the second set specifies the min and max value of rating each item *you can change this to change size of values which changes sizes of your points in this case. 
    # we then apply this to the matrix of choice. 
sz= sz(deg) ## in matlab this is done in function but here we have to apply the interp1d object to the degrees matrix

# figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sctt=ax.scatter3D(coor[0],coor[1],coor[2], c=cicon, s=sz/4) ## put the values here to plot
#fig.colorbar(sctt, ax = ax) ## adding colorbar 
  ax.view_init(100, -90)
  #ax.invert_yaxis()
plt.show() plt.clf()




                  #### Section 2: communities in functional networks ####

labels = pd.unique(system_labels[0])

#  to estimate communities, we run the louvain algorithm. We need to repeat
  # the algorithm multiple times due to its non-deterministic nature.
num_nodes   = np.size(cij,1)
num_reps    = 5


# gamma a range list requires a function in python
def range_inc(start, stop, step, inc):
    i = start
    while i < stop:
        yield i
        i += step
        step += inc

gamma_range = list(range_inc(0.5, 4.25, 0.25, 0)) 
    ## the reason I have the stop at 4.25 is becaues I Want to include 4 at the end
num_gamma   = len(gamma_range)



# preallocate some arrays to store communities and similarity scores
ci = np.zeros([num_nodes,num_reps,num_gamma]);
d = np.empty([num_reps,num_gamma])
d[:] = np.NaN 


  ## to adjust values to all positive so the louvain will work

from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,num_gamma):
    gamma = gamma_range[i]
    for ii in range(0,num_reps):
      print(ii)
      ci[:,ii,i] = bct.community_louvain( np.array(cij),gamma,'negative_asym')[0] ## negative_asym is to deal with negative values
      d[ii,i] = adjusted_rand_score(np.array(system_labels[0]),ci[:,ii,i])

  # finding the value at gamma zrand is peaked
    # findign values for each
peaks =[]
for i in d:
  peaks.append(max(i))

    # peak of all max values    
peak_gamma = max(peaks)
num_reps = 100
ci       = np.zeros([num_nodes,num_reps]);

with parallel_backend('threading', n_jobs=12): 
  for i in range(0,num_reps):
    ci[:,i] = bct.community_louvain(abs(np.array(cij)),gamma)[0]

# calculate consensus communities 
thr = 0.5
with parallel_backend('threading', n_jobs=12): 
  d = bct.agreement(ci)/num_reps

with parallel_backend('threading', n_jobs=12): 
  cicon = bct.consensus_und(d,thr,num_reps)


## Yet to do - plotting 


