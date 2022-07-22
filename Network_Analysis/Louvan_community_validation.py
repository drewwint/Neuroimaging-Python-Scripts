###############################################################################
### Code for validating Louvain Algorithm Community Detection #################
### For Multiple Participants and estimating each individually ################
### Code by: Drew E. Winters, PhD. ############################################
###############################################################################

# the logic
 #> the Louvain algorithm is stochastic so we need to use a principled approach
    #> to produce reliable result with this algorithm
 #> here we 
    #> 1) hyperparameter tune the gamma parameter  
        #> we do this over estimating 5 communities for each individual
    #> 2) re-estimating 5 individual-level communities with tuned gamma
    #> 3) calculate simiilarity between the 5 individual-level communities
    #> 4) derive the consensus commuity for each individual. 


num_nodes   = np.size(prec_mat[1],1);
num_participants = len(prec_mat)
num_reps    = 5;


# gamma a range list
def range_inc(start, stop, step, inc):
    i = start
    while i < stop:
        yield i
        i += step
        step += inc

gamma_range = list(range_inc(0.5, 4.25, 0.25, 0)) 


num_gamma   = len(gamma_range);


labels = pd.read_csv(r"C:\Users\wintersd\OneDrive - The University of Colorado Denver\1 Publications\CU traits and ANTS cog funct\Subj_timeseries_denoised\ROInames.csv", header=None).iloc[:,3:167]
labels = np.array(labels)[0]



from sklearn.metrics import rand_score

cia = []
da = []
# 
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for z in range(0,num_participants): 
    # preallocate some arrays to store communities and similarity scores
    ci = np.zeros([num_nodes,num_reps,num_gamma]);
    d = np.empty([num_reps,num_gamma])
    d[:] = np.NaN 
    for i in range(0,num_gamma):
      gamma = gamma_range[i]
      for ii in range(0,num_reps):
        print(ii)
        ci[:,ii,i] = bct.community_louvain((prec_mat[z]),gamma = gamma, B='negative_asym')[0]
        d[ii,i] = rand_score(ci[:,ii,i],np.array(labels))**2## thsi is the zrand score
    cia.append(ci)
    da.append(d)




b= []
a=np.zeros([len(da),len(da[0])])
for i in range(0, len(da)):
  for ii in range(0,len(da[i])):
    a[i][ii]=(max(da[i][ii]))
  b.append(max(a[i]))


## identify the gamma at which similarity has peaked for each participant 
ind_gamma = []
for i in range(0, len(da)):
  if int(np.where(np.matrix.flatten(da[i]) == b[i])[0]) >= 15:
    ind_gamma.append(gamma_range[int(round(int(np.where(np.matrix.flatten(da[0]) == b[0])[0])%15))])
  else: 
    ind_gamma.append(gamma_range[int(np.where(np.matrix.flatten(da[i]) == b[i])[0])])





## rerunning community detection with the optimal gamma value
num_reps = len(labels)

ddu = []
for i in range(0,num_participants):
  aa = []
  for ii in range(0, num_reps):
    aa.append(bct.community_louvain(prec_mat[i], gamma = ind_gamma[i], B='negative_asym')[0])
  ddu.append(aa)


# calculating consensus communities 
  # threshold for consensus 
thr = 0.5

d = []
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(ddu)):
    d.append(bct.agreement(ddu[i])/num_reps)

cicon = []
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=12): 
  for i in range(0,len(d)):
    cicon.append(bct.consensus_und(d[i],thr,num_reps))

# cicon is the list of one consensus community for each individual.



