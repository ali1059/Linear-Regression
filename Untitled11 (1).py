import matplotlib.pyplot as plt
import numpy as np
my_data = np.genfromtxt('dataa.csv', delimiter=',')
X = my_data[:, 0].reshape(-1,1) # -1 tells numpy to figure out the dimension by itself
ones = np.ones([X.shape[0], 1]) # create a array containing only ones 
X = np.concatenate([ones, X],1) # cocatenate the ones to X matrix
y = my_data[:, 1].reshape(-1,1)
plt.scatter(my_data[:, 0].reshape(-1,1), y)
plt.show()

alpha = 0.0005
iters = 10000

# theta is a row vector
theta = np.array([0,0])

theta=theta.reshape(-1,1)
print(theta)

#Compute Cost
def costfun(X,y,theta):
    z=np.power((np.dot(X,theta)-y),2)
    s=np.sum(z) / (2 * len(X))
    return s


def Gradient(X,y,theta,alpha,iters):
    for i in range(iters):
        theta=theta- (alpha/len(X)) * np.sum((np.dot(X,theta)-y)*X)
        cost=costfun(X,y,theta)
        
    return (theta,cost)

g=Gradient(X,y,theta,alpha,iters)

plt.scatter(my_data[:, 0].reshape(-1,1), y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim()) 
y_vals = g[0][0] + g[0][1]* x_vals #the line equation
plt.plot(x_vals, y_vals, '--')
plt.show()
