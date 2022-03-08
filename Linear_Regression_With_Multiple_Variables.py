import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

#======================= feature normalization Function =======================
def featureNormalize(x):
    x_norm =x
    mu = np.zeros([1,x.shape[1]])
    sigma = np.zeros([1,x.shape[1]])
    for i in range(0,x.shape[1]):
        mu[:,i]=x[:,i].mean()
        sigma[:,i]=x[:,i].std()
        x_norm[:,i]=(x[:,i]-mu[:,i])/sigma[:,i]
    return x_norm,mu,sigma
#======================= Cost Function for multi variables=======================
def computeCostMulti(x, y, theta):
    m = len(y)
    prediction = np.dot(x, theta)
    j = (sum(np.square(prediction - y))) / (2 * m)
    return j
#======================= Gradient Descent =======================
def gradientDescentMulti(x, y, theta, alpha, num_iters):
     m = len(y)
     j_history = np.zeros(num_iters)
     for i in range(0,num_iters):
         prediction = np.dot(x, theta)
         theta=theta-(alpha/m)*np.dot(x.transpose(),(prediction-y))
         j_history[i]=computeCostMulti(x,y,theta)
     return (theta, j_history)

#======================= Normal equation =======================
def normal_equation(x,y):
    theta=np.dot(inv(np.dot(x.transpose(),x)),np.dot(x.T,y))
    return theta


#======================= load data =======================

data=np.loadtxt('ex1data2.txt',delimiter=',')
X=np.c_[data[:,[0,1]]]
y=np.c_[data[:,2]]

#======================= normalize features =======================

X,mu,sigma=featureNormalize(X)
# Add X0 to Xs
X=np.insert(X,0,np.ones(X.shape[0]),axis=1)
#print(X,'\n',mu,'\n',sigma)

#============Init Theta and Run Gradient Descent ===========

theta=np.zeros([3,1])
alpha = 0.01
num_iters = 400
theta , Cost_J=gradientDescentMulti(X,y,theta,alpha,num_iters)
print('optimal theta after running Gradient Descent : ',theta,'\n')

# Plot the convergence graph

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()

#test

House_size=(1650-mu[:,0])/sigma[:,0]
num_of_bedrooms=(3-mu[:,1])/sigma[:,1]
predicted_price=np.dot(np.matrix([[1, House_size, num_of_bedrooms]],dtype=object),theta)
#predicted_price=np.dot(np.array([[1,(1650-mu[:,0])/sigma[:,0],(3-mu[:,1])/sigma[:,1]]],dtype=object),theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) :',predicted_price,'\n')

#============Calculate the parameters from the normal equation ===========
theta=normal_equation(X,y)
print('optimal theta after running normal equation : ',theta,'\n')

#Test
predicted_price=np.dot(np.matrix([[1, House_size, num_of_bedrooms]],dtype=object),theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equation) :',predicted_price )