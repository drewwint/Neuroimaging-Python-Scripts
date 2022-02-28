
# Predictive Modeling - 1
# Ridge and Lasso
  #> in this lab we are going to numerically explore the process of using 
    #> Ridge Regression and Lasso Regression to solve prediction problems that 
    #> otherwise would be unsolvable using Ordinary Least Squares (OLS).
  #> Before we hit Ridge and Lasso, let us revisit OLS briefly in the context 
    #> of predictive modeling, and investigate how quantity of training data 
    #> influences the quality of the prediction.
  #> 



import matplotlib.pyplot as plt
import seaborn as sns   #
import pandas as pd     #  
import numpy as np      # 
import networkx as nx   # for network analysis
import bct              # brain connectivity toolbox






                      #### OLS Regression ####



# plotting initial scatter
plt.scatter(xtrain,ytrain, c='r', s=100),  plt.show(), plt.clf()

# setting up regresison line 
  # x design matrix
    # here we are concatenating a list of 1's for the intercept with the xtrain dataframe... that simple
x =pd.concat([pd.DataFrame(np.matrix(np.ones(np.size(xtrain,0))).T),xtrain],axis=1)

  
b_hat = np.linalg.inv(x.T @ x) @ x.T @ ytrain
  # thsi is the formula for a regression line the produces an intercept and slope (beta) parameter
    # thsi is produced becasue we fed it the list of 1's for intercept and x values for slope
inter = b_hat[0][0]
b_hat2 = b_hat[0][1] 



plt.scatter(xtrain,ytrain, c="r",s=100), plt.plot(xtrain,(b_hat2*xtrain + inter) ,c="b") , plt.show(), plt.clf()


  # calculating yhat - I dint need to do thsi to plot the line becuase I alreadi specified the value above
y_hat = x @ np.array(b_hat)
    ## this approach produces the same results BUT:
    ## for some reason - when I put this in the plot it doenst produc the line 
    ## but when I put whet I put above in the plot it creates a line 



# now we will explore how the quantity of trainign data impacts the quality of resulting data fit


  # Specify a fraction to use.
fractionDataTrain = .05;

  # Need to do a small calculation for internal use of the cvpartition command.
fractionData = 1 - fractionDataTrain;

  # Make sure there is some data, but not more than fraction 1
fractionData = min(1,max(fractionData,.05));


  # making traning sets of data
    ## note in order to get the same # of train datasets as matlab I need to do 0.94 instead of the fractionData
from sklearn.model_selection import train_test_split

xtrain, xtest = train_test_split(XTrainData, test_size=0.94, random_state=42)

ytrain, ytest = train_test_split(YTrainData, test_size=0.94, random_state=42)

  # setting up the same plot wiht this training data
    # setting up regresison line 
    # x design matrix
      # here we are concatenating a list of 1's for the intercept with the xtrain dataframe... that simple
x =pd.concat([pd.DataFrame(np.matrix(np.ones(np.size(xtrain,0))).T,index=[20,60,71,14,92,51]),xtrain],axis=1)

  
b_hat = np.linalg.inv(x.T @ x) @ x.T @ ytrain
  # thsi is the formula for a regression line the produces an intercept and slope (beta) parameter
    # thsi is produced becasue we fed it the list of 1's for intercept and x values for slope
inter = b_hat[0][0]
b_hat2 = b_hat[0][1] 



plt.scatter(xtrain,ytrain, c="r",s=100), plt.plot(xtrain,(b_hat2*xtrain + inter) ,c="b") , plt.show(), plt.clf()


  # calculating RMSE
xx = pd.concat([pd.DataFrame(np.matrix(np.ones(np.size(XTestData,0))).T),XTestData],axis=1)
xxx = pd.concat([pd.DataFrame(np.matrix(np.ones(np.size(XTrainData,0))).T),XTrainData],axis=1)
np.sqrt(np.mean((YTrainData - xx @ np.array(b_hat))**2))
np.sqrt(np.mean((YTestData - xxx @ np.array(b_hat))**2))


# plotting 
plt.scatter(xtrain,ytrain ,label="Data", c="red",s=50), 
  plt.plot(xtrain,(b_hat2*xtrain + inter),label="OLS Fit", c="red"),
  plt.legend(loc="upper left")
  plt.text(2.5,11,"$RMSE_{Train} = $" + str(round(float(np.sqrt(np.mean((YTestData - xx @ np.array(b_hat))**2))),3)))
  plt.show(), plt.clf(), plt.cla()

plt.scatter(xtrain,ytrain,label="Train Data", c="red",s=50), 
  plt.plot(xtrain,(b_hat2*xtrain + inter),label="OLS Fit", c="red"),
  plt.scatter(XTestData,YTestData, label= "Test Data" , c="black" , s=50), 
  plt.legend(loc="upper left")
  plt.text(2.5,6.5,"$RMSE_{Train} = $" + str(round(float(np.sqrt(np.mean((YTrainData - xx @ np.array(b_hat))**2))),2))),
  plt.text(2.5,5,"$RMSE_{Test}  = $" + str(round(float(np.sqrt(np.mean((YTestData - xxx @ np.array(b_hat))**2))),2)),c="r")
  plt.show(), plt.clf(), plt.cla()






                     #### Ridge Regression ####
    #> A critical aspect of OLS is the need to calculate the inversion of the matrix
    #> But there can be situations where this matrix is badly conditioned 
      #> (meaning that it is not really invertable), which can happen if we have 
      #> a general linear model in which two or more predictors are highly correlated. 
    #> This can happen quite often in functional MRI designs if care is not taken, 
      #> or can occur when clinical measures might be highly correlated. 
    #> It is also possible for X.TX to not be invertable if we do not have 
      #> sufficient number of observations in our training dataset. 
    #> As we saw above, in the OLS toy models, there can be times when the 
      #> training dataset might be insufficient, which can result in wildly 
      #> varying quality of prediction. A means to address the issue of badly 
      #> conditioned models is to use what is known as regularization. 
      #> Mathematically, we can address the instabilities of a badly 
      #> conditioned design by the introduction of a penalty factor.
          
  ## a helpful tutorial on ridge regression in python
    #> https://jbhender.github.io/Stats506/F17/Projects/G13/Python.html 


cond1=np.linalg.cond(XTrain.T @ XTrain)

# creating a design matrix
xx = pd.concat([pd.DataFrame(np.matrix(np.ones(np.size(YTrain,0))).T),XTrain],axis=1)


b_hat = np.linalg.inv(xx.T @ xx) @ (xx.T @ YTrain)

# Visualize Cross Correlatoin
  # similar results of both of these
sns.heatmap(np.corrcoef(XTrain.T), cmap="jet"), plt.show(), plt.clf(), plt.cla()
  # this one is shrinked smaller than the axsis for some reason. 
sns.heatmap(np.corrcoef(xx.T), cmap="jet"), plt.show(), plt.clf(), plt.cla()


# building identity matrix
IMatrix = np.eye(np.size(XTrain,1))

sns.heatmap(IMatrix, cmap ="viridis"), plt.show(), plt.clf(), plt.cla()



# Defininig lambda - the penalty value
Lambda = .01
# building the regularized design matrix
XRidge = pd.DataFrame(np.vstack([XTrain,np.multiply(Lambda, IMatrix)]))
# Adding some bogus observations to our data - the same number of 0's as teh matrix XTrain
YRidge = np.vstack([YTrain, np.matrix(np.zeros(np.size(XTrain,1))).T])


# calculate condition number
cond2=np.linalg.cond(XRidge.T @ XRidge)
print("condition 1 = ", cond1, "condition 2 = ", cond2)
  ## not a small number but smaller than the condition calculated above. 




# ridge regression
  ## we coudl just pick a random lambda but to obtimize the penality we can split our datset into a training and test
  
  ## They already split the data above with X/YTrain (for train) and X/YTune

from sklearn.linear_model import Ridge
# not used: from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
  # or could do this and youd have to referr to sklearn.metrics.mean_squared_error()
  # import sklearn.metrics

# lambda ridge penalty
ks = np.arange(0,0.2005,0.005)

#initialize an array for RMSE
rmsErrs = np.array([np.zeros(np.size(ks))]).T

# initialize list for coefficients
coef = []

# Figure plotting each individual lambda
  #> this figure runs a ridge regression
  #> then appends the coefficient to a list
  #> calculates YHat
  #> appends to the list of rmse
  #> plots the datapoints and regression line for each lambda
for i in range(0,np.size(ks)):
  k = ks[i]
  ridge = Ridge(alpha=k)
  ridge.fit(XTrain,YTrain)
  coef.append(ridge.coef_)
  YHat= ridge.predict(np.array(XTune))
  rmsErrs[i] = mean_squared_error(YTune,YHat,squared=False)
  plt.scatter(YTune,YHat, facecolors='none',edgecolors='b',alpha=.35), 
  plt.plot(YTune,YTune, c= "r"), 
  plt.text(4.25,2.5,"RMSE = " + str(round(mean_squared_error(YTune,YHat,squared=False),3)), c="r"),
  plt.text(4.25,3,r"$\lambda = $" + str(round(k,3)), c="r"),
  plt.show(), plt.clf(), plt.cla()

# additional plots
  #Plotting ridge coefficients vs regularization parameters
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(ks,np.array(coef)[:,0,:])
ax.set_xlabel("Ridge Penalty " + r"$\lambda$")
ax.set_ylabel( "Scaled " + r"$\beta$'s")
ax.set_title('Ridge Regression Optimization')
ax.axis('tight')
plt.show(), plt.clf(), plt.cla()

  #RMSE plot 
plt.plot(ks,rmsErrs), plt.ylabel("RMSE"), plt.xlabel( "Ridge Penalty " + r"$\lambda$") ,plt.show(), plt.clf(), plt.cla()


  ## putting the two plots together
fig, axes = plt.subplots(2)
fig.suptitle('Ridge Regression Optimization')
fig.supxlabel("Ridge Penalty " + r"$\lambda$")
axes[0].plot(ks,np.array(coef)[:,0,:])
axes[1].plot(ks,rmsErrs)
axes[0].set_ylabel( "Scaled " + r"$\beta$'s")
axes[1].set_ylabel("RMSE")
plt.show(), plt.clf(), plt.cla()



# Identifying the optimal lambda parameter
  # to do so we will use Cross Validation 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

scaler = StandardScaler()

X_std = scaler.fit_transform(XTrain)

regr_cv=RidgeCV(alphas=ks[1:]) 
model_cv=regr_cv.fit(X_std,YTrain)
print(model_cv.alpha_)
  # the optimal lambda here is 0.2




# plotting the optimal ridge

Ridge(alpha=model_cv.alpha_)
ridge.fit(XTrain,YTrain)
YHat= ridge.predict(np.array(XTune))
RMSE = mean_squared_error(YTune,YHat,squared=False)
plt.scatter(YTune,YHat, facecolors='none',edgecolors='b',alpha=.35), 
plt.plot(YTune,YTune, c= "r"), 
plt.text(4.25,2.5,"RMSE = " + str(round(mean_squared_error(YTune,YHat,squared=False),3)), c="r"),
plt.text(4.25,2.8,r"$\lambda = $" + str(round(model_cv.alpha_,3)), c="r"),
plt.title("Optimal Ridge Regression")
plt.ylabel("Predicted Observation")
plt.xlabel("Actual Observation")
plt.show(), plt.clf(), plt.cla()




# Testing our model on the test data
YPred = ridge.predict(XFittest)
testRMSE = mean_squared_error(YPred,YFittest,squared=False)
testRMSE

plt.scatter(YFittest,YPred, facecolors='none',edgecolors='b',alpha=.35), 
plt.plot(YFittest,YFittest, c= "r"), 
plt.text(4.25,2.5,"RMSE = " + str(round(testRMSE,3)), c="r"),
plt.text(4.25,2.7,r"$\lambda = $" + str(round(model_cv.alpha_,3)), c="r"),
plt.title("Test Data Optimal Ridge Regression")
plt.ylabel("Test Predicted Observation")
plt.xlabel("Test Actual Observation")
plt.show(), plt.clf(), plt.cla()






              #### Lasso Regression #####
  # lasso is similar to ridge regression only the penalty is different 
    #> while Ridge Regression will penalize all parameters, 
      #> Lasso can result in driving some parameters to zero, 
      #> resulting in what is called a sparse solution
    #> The cost function for Lasso is, where now the penalty is linear in beta 
      #> (what is known as the norm, while Ridge uses the norm): 

from sklearn import linear_model
from sklearn.linear_model import LassoCV

# since we will do cross validation - go ahead and put train and tune DFs together
XBunch = pd.concat([XTrain,XTune])
YBunch = pd.concat([YTrain,YTune])

las = LassoCV(cv=10).fit(XBunch,np.ravel(YBunch)) # had to use ravel to flatten the column vector to run

las.score(XBunch,YBunch)
las.predict(XBunch)
las.coef_
las.intercept_
las.alphas_


Ylas=las.predict(XFittest)
RMSElas = mean_squared_error(Ylas,YFittest,squared=False)
RMSElas



plt.scatter(YFittest,Ylas, facecolors='none',edgecolors='b',alpha=.35), 
plt.plot(YFittest,YFittest, c= "r"), 
plt.text(4.25,2.5,"RMSE = " + str(round(RMSElas,3)), c="r"),
plt.text(4.25,2.7,r"$\lambda = $" + str(round(las.alpha_,3)), c="r"),
plt.title("Test Data Optimal Lasso Regression")
plt.ylabel("Test Predicted Observation")
plt.xlabel("Test Actual Observation")
plt.show(), plt.clf(), plt.cla()



# still to do - create lasso plot like matlab
  # Same as lassoPlot in matlab
    # cite: https://sprjg.github.io/posts/lassoplot_in_python/ 



















