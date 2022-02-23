

# Relating Structure and Function throung communication models 


import matplotlib.pyplot as plt
import seaborn as sns   #
import pandas as pd     #  
import numpy as np      # 
import networkx as nx   # for network analysis
import bct              # brain connectivity toolbox

from scipy.spatial.distance import pdist, squareform

## calculating the euclidean distance between nodes
  ## not sure when but will use it later
euc = squareform(pdist(coor))


# Binsarizing the connectiviy matrix and calculating distance
thr = 0.01;
SCbin = (bct.threshold_proportional(np.array(sc),thr) > 0).astype(int);
  # Let's calculate the binary shorest path matrix
d = bct.distance_bin(SCbin);



num_nodes = len(sc);
ut_mask = (np.triu(np.ones(num_nodes),1) > 0).astype(int);

# Set up variables for regression
X = np.array([np.ones(np.count_nonzero(ut_mask)),d[np.nonzero(ut_mask)]]);
#len(X[0])

y = np.array(np.matrix(fc)[np.nonzero(ut_mask)])[0] 
    ## note here I have to make the fc a matrix inorder to refer to the cells appropriately
    ## then transform it back into an array 
    ## then indicate a [0] at the end so that I can get rid of one pair of [] so it is easier to access

# regression 
import statsmodels.api as sm
import statsmodels.formula.api as smf
  ## To do this I created a dataframe from the arrays created (by using a dict to do this)
  ## then use the regression method using stats models

a=pd.DataFrame({"X1"X[0],"X2"X[1],"y"y},index=range(0,len(X[0])))
model = smf.ols(formula="y~X2", data=a).fit() 
print(model.summary())
beta = model.params
variance_exp = model.rsquared


# making the best fit line 
xfit= np.ones(2),[min(d[np.nonzero(ut_mask)])-1,max(d[np.nonzero(ut_mask)])+1]
  #yfit = np.matrix([[1,1],[0,12]]).T @ beta
yfit = np.vstack([np.matrix(np.ones(2)),np.matrix([min(d[np.nonzero(ut_mask)])-1,max(d[np.nonzero(ut_mask)])+1])]).T @ beta


# plotting 
plt.scatter(d[np.nonzero(ut_mask)],y, c='black', alpha=.05)
  plt.plot(xfit[1],yfit, c="r")
  plt.xlim(0,11.5)
  plt.ylim(-0.4,0.8)
  plt.text(10,.7,"$r^2 = $" + str(round(variance_exp,2))) ## note how I specify the superscript here
  plt.xlabel("path length, number of steps")
  plt.ylabel("$FC, r_{z}$") ## note how I denote tye subscript here
  plt.show(), plt.clf()


  ## same plot just with jitter
jittered_x = d[np.nonzero(ut_mask)] + 0.2 * np.random.rand(len(d[np.nonzero(ut_mask)])) -0.05


plt.scatter(jittered_x,y, c='black', alpha=.03)
  plt.plot(xfit[1],yfit, c="r")
  plt.xlim(0,11.5)
  plt.ylim(-0.4,0.8)
  plt.text(10,.7,"$r^2 = $" + str(round(variance_exp,2))) ## note how I specify the superscript here
  plt.xlabel("path length, number of steps")
  plt.ylabel("$FC, r_{z}$") ## note how I denote tye subscript here
  plt.show(), plt.clf()





gamma = np.arange(0.1,2.1,0.1);
variance_explained_wei = np.zeros(len(gamma));
variance_explained_hops = variance_explained_wei;


  ## transforming the structural connectivyt matrix into a "cost" matrix

variance_explained_wei2 = []
variance_explained_hops2 = []

for i in range(0,len(gamma))
  L = sc ** -gamma[i]
  dwei,dhops,aa = bct.distance_wei_floyd(np.array(L))
  X1 = np.array([np.ones(np.count_nonzero(ut_mask)),dwei[np.nonzero(ut_mask)]]);
  y1 = np.array(np.matrix(fc)[np.nonzero(ut_mask)])[0]
  b=pd.DataFrame({"X1"X1[0],"X2"X1[1],"y"y1},index=range(0,len(X[0])))
  model1 = smf.ols(formula="y~X2", data=b).fit() ## I only specify X2 b/c this command already calculates the intercept
  variance_explained_wei2.append(model1.rsquared)
  X2 = np.array([np.ones(np.count_nonzero(ut_mask)),dhops[np.nonzero(ut_mask)]]);
  y2 = np.array(np.matrix(fc)[np.nonzero(ut_mask)])[0]
  c=pd.DataFrame({"X1"X2[0],"X2"X2[1],"y"y2},index=range(0,len(X[0])))
  model2 = smf.ols(formula="y~X2", data=c).fit()
  variance_explained_hops2.append(model2.rsquared)
  


  # comparing the vars, dhops looks fine but dewei needs divided by 10000 for the correct values
dwei= dwei/100000




#barplot of variance explained by shortest paths
sns.barplot(x=gamma,y=variance_explained_wei2, color="blue"), 
  plt.xticks(ticks=np.arange(20),labels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
  plt.xlabel(r'$\gamma$ values'), 
  plt.ylabel('variance explained'), 
  plt.ylim(0,0.06), 
  plt.text(1.7,0.049,"*",c="r",fontsize=30)
  plt.show(), plt.clf()

# number of hops
sns.barplot(x=gamma,y=variance_explained_hops2, color="blue"), 
  plt.xticks(ticks=np.arange(20),labels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
  plt.xlabel(r'$\gamma$ values'),
  plt.ylabel('variance explained'), 
  plt.ylim(0,0.07), 
  plt.text(12.7,0.0595,"*",c="r",fontsize=30)
  plt.show(), plt.clf()




                    #### Part 2 Alternative communication models and model comparison ####


# Binarizing the connectiviyt matrix and calculating distance 
g= nx.Graph(toy_dt)
p= nx.shortest_path(g,source=0,target=8)
k=np.sum(toy_dt,1)
pk = 1/k[p]
si= -np.nansum(np.log2(pk))


  ## you could add another connection like ths and recalculate
CIJ= np.matrix(toy_dt)

CIJ[1,6] = 1
CIJ[6,1] = 1
g= nx.Graph(CIJ)
#p= nx.shortest_path(g,source=0,target=8)
k=np.sum(CIJ,1)
pk = 1/k[p]
si= -np.nansum(np.log2(pk))
si ## note the si increased from the previous 5.1699


  ## you could remove a connection 
CIJ= np.matrix(toy_dt)

CIJ[1,2] = 0
CIJ[2,1] = 0
g= nx.Graph(CIJ)
#p= nx.shortest_path(g,source=0,target=8)
k=np.sum(CIJ,1)
pk = 1/k[p]
si= -np.nansum(np.log2(pk))
si # note this decreased from the origional 5.1699



#  To speed things up, we'll focus just on intra-hemispheric connectivity

idxhemi = np.concatenate([np.ones(108).astype(int), (np.zeros(111)).astype(int)]);

SC = np.matrix(sc)[np.nonzero(idxhemi)]

FC = np.matrix(fc)[np.nonzero(idxhemi)]
pd.DataFrame(FC) ## the top portion lines up well but not so sure about the bottom portion 

## still to do - predict FC and plot 


