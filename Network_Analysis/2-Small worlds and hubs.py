

# Small worlds and hubs 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import networkx as nx # for network analysis
import bct # brain connectivity toolbox
  ## note info on brian connectivyt toolbox for python can be found here
    # https://github.com/aestrivex/bctpy 



                  #### Network Basics ####


# binarizing the matrix
CIJbin = (cij > 0).astype(int)

# calculate clustering coefficient.
c_region = bct.clustering_coef_bu(np.array(CIJbin))
C = np.mean(c_region)

# next, we need to calculate the network's path length. just last time,
    # this involves first calculating the matrix of distances between each pair of regions.
d = bct.distance_bin(np.array(CIJbin))

# then we pass this variable to the 'charpath' function
  # charpath calculates
    # lambda = L : 
        #>  The network characteristic path length is the average shortest path 
        #>  length between all pairs of nodes in the network
    # global efficiency :
        #> The global efficiency is the average inverse shortest path length in the network.
    # nodal eccenctircity  = ecc :
        #> The nodal eccentricity is the maximal path length between a node and any other node in the network
    # radius : 
        #> The radius is the minimal eccentricity
    # diameter :
        #> diameter is the maximal eccentricity 
L,efficiency,ecc,radius,diameter = bct.charpath(d) #[0] I could also just speciy the first element like this
L


# the variables C and L on their own tell us very little. we need to
          # compare them against an ensemble of random networks.
          # set some parameters
Num_samples = 1
Itera = 2**5



# generate a new randomized network each time thoruhg the loop 

  # initalizing lists to appent to
L_rand = []
C_rand = []

for i in range(0,Num_samples):
  CIJbin_rand = bct.randmio_dir(np.array(CIJbin), Itera)
  D_rand = bct.distance_bin(CIJbin_rand[0])
  L_rand.append(bct.charpath(D_rand)[0])  
  c_region_rand = bct.clustering_coef_bu(CIJbin_rand[0])
  C_rand.append(np.mean(c_region_rand))


## normalizing clustering coefficient and characteristic path length
C_norm = C/np.mean(C_rand)
L_norm = L/np.mean(L_rand)

  # these are cluater and length normalized values






          #### Hubs ####


# Calculate degrees and strenghts
deg = bct.degrees_und(cij)
stre = bct.strengths_und(cij)




# plotting degree & strength relatoinship

plt.scatter(deg,stre) 
  plt.text(70,0,"r = " + str(round(np.corrcoef(deg,stre)[0,1],2)))
  plt.xlabel("degree")
  plt.ylabel("strength")plt.show() plt.clf()





    ### CALCULATING BINARY AND WEIGHTED BETWEENNESS CENTRALITY###

# We can calculate betweenness centrality easily for binary networks
CIJ_bin = (cij > 0).astype(int)
btwn_bin = bct.betweenness_bin(CIJ_bin)


# Taking the reciprocal of each edge's weight
gamma = 1
Cost = cij**(-gamma)
Cost[np.isinf(Cost)] = 0  ## recoding inf to 0's
# Note: next step will take a minute to run on the virtualbox
btwn_wei = bct.betweenness_wei(np.array(Cost))
N = len(cij)
btwn_wei = btwn_wei/((N - 1)*(N - 2))

## not that we have bianary/weighted degree and betweeness centrality 
  ## we can combine them into a composite score



import scipy 
rank_matrix = scipy.stats.rankdata([deg,stre,btwn_bin,btwn_wei], axis = 1)/1000
  ## note - I divide by 1000 and I get the exact same number as matlab :) 

ave_rank = np.mean(rank_matrix,axis=0)*1000
  ## note I times by 1000 and I get the same values as matlab...? 
  

## Plotting Hubs

  ## setting up a 3d figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sctt=ax.scatter3D(coor[0],coor[1],coor[2], c=ave_rank, s=ave_rank/15) ## put the values here to plot
fig.colorbar(sctt, ax = ax) ## adding colorbar 
  #ax.view_init(-130, -150)
  #ax.invert_yaxis()
  #ax.invert_xaxis()
  #ax.invert_zaxis()
plt.show() plt.clf()





