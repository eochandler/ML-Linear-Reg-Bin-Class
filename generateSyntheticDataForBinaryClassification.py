import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.linear_model

def plotLine(beta,xMin,xMax,yMin,yMax):
    # This function plots the decision boundary determined by vector beta = [beta0,beta1,beta2]
    
    xVals = np.linspace(xMin,xMax,100)
    yVals = (-beta[1]*xVals - beta[0])/beta[2]
    
    idxs = \
        np.where((yVals >= yMin) & (yVals <= yMax))
    
    plt.plot(xVals[idxs],yVals[idxs])

# Set up some synthetic data to test the logistic regression algorithm on.
numPos = 100
numNeg = 100

np.random.seed(7) # Set random number generator seed so that results are reproducible.
muPos = [1.0,1.0]
covPos = np.array([[1.0,0.0],[0.0,1.0]])

muNeg = [-1.0,-1.0]
covNeg = np.array([[1.0,0.0],[0.0,1.0]])

Xpos = np.zeros((numPos,2))

for i in range(numPos):
    Xpos[i,0:2] = \
        np.random.multivariate_normal(muPos,covPos)
        
Xneg = np.zeros((numNeg,2))
for i in range(numNeg):
    Xneg[i,0:2] = \
        np.random.multivariate_normal(muNeg,covNeg)
        
X = np.concatenate((Xpos,Xneg)) # X has one row for each training example
Y = np.zeros(numPos + numNeg) 
Y[0:numPos] = 1 

def crossEntropy(p,q):
    return -p*np.log(q) - (1-p) * np.log(1-q)

def sigmoid(u):
    return np.exp(u)/(1+ np.exp(u))

def h(u, yi):
    exp = np.exp(u)
    return -yi * u + np.log(1 + exp)

def L(beta, X, Y):
    N = X.shape[0]
    mySumHi = 0
    for i in range(N):
        xihat = X[i]
        yi = Y[i]
        dotProduct = np.vdot(xihat, beta)
        mySumHi += h(dotProduct, yi)
    return mySumHi

def logReg(X, Y):
    N, d = X.shape
    allOnes = np.ones((N, 1))
    X = np.hstack((allOnes, X))
    
    alpha = 0.001
    beta = np.random.randn(d+1)
    listNormGrad = []
    listLBetas = []
    
    maxiterations = 1000
    plt.figure()
    for i in range(maxiterations):
        gradient = np.zeros(d+1)
        
        for i in range(N):
            Xi = X[i, :]
            Yi = Y[i]
            qi = sigmoid(np.vdot(Xi, beta))
            
            gradient += (qi - Yi) * Xi
            
            
        norm_gradient = np.linalg.norm(gradient)
        listNormGrad.append(norm_gradient)
        beta = beta - alpha * gradient  
        LBeta = L(beta, X, Y)
        listLBetas.append(LBeta)
        
    return beta, listNormGrad, listLBetas
xMin = min(X[:,0])
xMax = max(X[:,0])
yMin = min(X[:,1])
yMax = max(X[:,1])

beta, myListNormGrad, listLBetas = logReg(X,Y)
plt.figure()
plt.semilogy(myListNormGrad)
plt.semilogy(listLBetas)
# the L(Beta) vales is decreasing


plt.figure()

plt.scatter(Xpos[:,0], Xpos[:,1])
plt.scatter(Xneg[:,0], Xneg[:,1])
plotLine(beta,xMin, xMax, yMin, yMax)
plt.axis("Equal")
plt.pause(.05)

print(beta)

#%%
beta = np.random.randn(3)
# Now plot the data and the line beta[0] + beta[1]*x1 + beta[2]*x2 = 0.

xMin = min(X[:,0])
xMax = max(X[:,0])
yMin = min(X[:,1])
yMax = max(X[:,1])
plt.gcf().clear()
model = sk.linear_model.LogisticRegression(solver = 'lbfgs' , penalty = 'none')
model.fit(X,Y)

#getting coefficient values from the model
bias = model.intercept_
weights = model.coef_[0]
beta_sk = [bias, weights[0], weights[1]]
plt.scatter(Xpos[:,0], Xpos[:,1])
plt.scatter(Xneg[:,0], Xneg[:,1])
plotLine(beta_sk, xMin, xMax, yMin, yMax)
plt.axis("equal")

plt.scatter(Xpos[:,0],Xpos[:,1])
plt.scatter(Xneg[:,0],Xneg[:,1])
plotLine(beta,xMin,xMax,yMin,yMax)
plt.axis("equal")
plt.pause(.05)
