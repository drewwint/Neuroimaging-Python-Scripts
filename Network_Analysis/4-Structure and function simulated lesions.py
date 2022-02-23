

# Simulated nodal lesions

import matplotlib.pyplot as plt
import seaborn as sns   #
import pandas as pd     #  
import numpy as np      # 
import networkx as nx   # for network analysis
import bct              # brain connectivity toolbox


          #### Section 1 removing structural nodes #### 


#### SIMULATED NODAL LESIONS ON EFFICIENCY #### 
          

CIJbin = (cij > 0).astype(int)
d = bct.distance_bin(np.array(CIJbin))
eff = bct.charpath(d)[1] ## note I specify [1] becaue that is the efficiency measure


# creating lesion matrix outside of the loop 

# this function is simulating lesions across the entire brain
  #> by making a 0 across each column and row in the matrix
  #> then retesting each time what happens to the efficiency using charpath

num_nodes = len(CIJbin) 
eff_lesion = []

for i in range(0,num_nodes)
  print(i)
  CIJ_lesion = np.matrix(CIJbin)
  CIJ_lesion[i,]= np.zeros(num_nodes)
  CIJ_lesion[,i]= np.zeros([num_nodes,1]) # note I use ,1 here to denote a column vector 
  d = bct.distance_bin(np.array(CIJ_lesion)) 
  eff_lesion.append(bct.charpath(d)[1]) # note I am indicating [1] becaues that is the efficiency output. 

## calculating the change in efficiency for each
delta_eff = eff_lesion-eff

# setting up the 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sctt=ax.scatter3D(coor[0],coor[1],coor[2], c=delta_eff)
fig.colorbar(sctt, ax = ax) ## adding colorbar 
plt.show(); plt.clf()



#### SIMULATED LESIONS ON WEIGHTED MODULARITY ####

q=bct.community_louvain(np.array(cij))[1]

num_nodes = len(CIJbin) 
eff_lesion = []

for i in range(0,num_nodes)
  print(i)
  CIJ_lesion = np.matrix(CIJbin)
  CIJ_lesion[i,]= np.zeros(num_nodes)
  CIJ_lesion[,i]= np.zeros([num_nodes,1]) # note I use ,1 here to denote a column vector 
  q_lesion = bct.community_louvain(np.array(CIJ_lesion))[1]

# change in modularity 
delta_q = q_lesion - q

# figure 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sctt=ax.scatter3D(coor[0],coor[1],coor[2], c=delta_q) ## put the values here to plot
fig.colorbar(sctt, ax = ax) ## adding colorbar 
plt.show(); plt.clf()



              #### Section 2 simulating lesions by removing edges ####


#  For this exercise, we'll calculate binary efficiency
CIJbin = (cij > 0).astype(int);
d = bct.distance_bin(np.array(CIJbin));
eff = bct.charpath(d)[1];

#  Now, we'll loop over all nodes and systematically remove those nodes from the network
num_nodes = len(CIJbin);

u = np.argwhere(np.triu(CIJbin,k=1))[,0]
v = np.argwhere(np.triu(CIJbin,k=1))[,1]

num_edges = len(v)

eff_lesion_mat = np.matrix(np.zeros([num_nodes, num_nodes]))
delta_eff_mat = eff_lesion_mat

#delta_eff_mat[u[1],v[1]] = 1

for i in range(0,num_edges)
  print(i/num_edges)
  CIJ_lesion = np.matrix(CIJbin)
  CIJ_lesion[u[i],v[i]] = 0
  CIJ_lesion[v[i],u[i]] = 0
  d = bct.distance_bin(np.array(CIJ_lesion))
  eff_lesion_mat[u[i],v[i]] = bct.charpath(d)[1]
  delta_eff_mat[u[i],v[i]] = (eff_lesion_mat[u[i],v[i]] - eff)\


#plotting changes in efficiency in each simulated lesion

sns.heatmap((delta_eff_mat+delta_eff_mat.T), cmap="viridis", 
                             cbar_kws={'label' 'change in efficiency'},
                             vmin=-np.matrix.max((abs(delta_eff_mat[]))*.5),
                             vmax= np.matrix.max((abs(delta_eff_mat[])))*.5), 
                             plt.xlabel("Brain Regions"), 
                             plt.ylabel("Brain Regions")
                             plt.show(), plt.clf()



## which connections lead to greater decremetns in dfficiency when lesioned intro or inter modular? 

niter= 100
ci= np.matrix(np.zeros([num_nodes,niter]))


for i in range(0,niter)
  ci[,i] = (np.matrix(bct.community_louvain(np.array(CIJbin))[0]).T)


thr = 0.5
cicon = bct.consensus_und(bct.agreement(ci)/int(niter),int(thr),int(niter))
type(cicon)

mask = np.triu(CIJbin)
d = bct.agreement(np.matrix(cicon).T)


d_msk = d[np.nonzero(mask)]

delta_msk = np.array(delta_eff_mat)[np.nonzero(mask)]

# t test
from scipy import stats
stats.ttest_ind(delta_msk,d_msk)
    ## OR
import researchpy as rp

a=pd.DataFrame({"d_msk"d_msk,"delta_msk"delta_msk})

b=rp.ttest(group1= a.delta_msk[a.d_msk==1], group1_name= "intra",
         group2= a.delta_msk[a.d_msk==0], group2_name= "inter")



# plot
ax=sns.boxplot(x=d_msk,y=delta_msk); 
  ax.set_xticklabels(["inter","intra"]); 
  plt.figtext(.6,.9,
    "T= " + str(round(np.array(np.matrix(b[1])[2])[0][1],2)) + 
    ", p =" +  str(np.array(np.matrix(b[1])[3,1])) + 
    ", " + str(np.array(np.matrix(b[1])[6,0])) + 
    str(np.array(round(np.matrix(b[1])[6,1],2))));
  plt.show(); plt.clf()


thr=0.01
mask = bct.threshold_proportional(abs(delta_eff_mat),thr)[bct.threshold_proportional(abs(delta_eff_mat),thr) !=0]


mask =  mask.reshape(2639,1).dot(delta_eff_mat[np.nonzero(delta_eff_mat)])
    
u,v = np.nonzero(np.triu(mask))
w= np.triu(mask)[u,v] # weights 


# setting up the fugure 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a=pd.DataFrame({"a"np.array(np.hstack(np.matrix(x_ln)))[0],"b"np.array(np.hstack(np.matrix(y_ln)))[0],"c"np.array(np.hstack(np.matrix(z_ln)))[0]})

a= a.drop_duplicates(["a","b","c"])

ax.plot3D(a.a,a.b,a.c, color= "b")


plt.show(); plt.clf()


