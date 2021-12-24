#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
data = pd.read_csv('/home/kato/fcai/ML/Assignment1/house_data.csv')
 
#show data details
print('data = \n' ,data.head(10) )
print('**************************************')
print('data.describe = \n',data.describe())
print('**************************************')

# rescaling data
data.grade = (data.grade - data.grade.mean()) / data.grade.std()
data.bathrooms = (data.bathrooms - data.bathrooms.mean()) / data.bathrooms.std()
data.lat = (data.lat - data.lat.mean()) / data.lat.std()
data.sqft_living = (data.sqft_living - data.sqft_living.mean()) / data.sqft_living.std()
data.view = (data.view - data.view.mean()) / data.view.std()


# separate X (training data) from y (target variable)
Xtrain = data.loc[:(len(data) * 80 / 100), ["grade","bathrooms","lat","sqft_living","view"]]
ytrain = data.loc[:(len(data) * 80 / 100),["price"]]

# add ones column
Xtrain.insert(0, 'Ones', 1)

# # separate X (test data) from y (target variable)
# Xtest = data.loc[(len(data) * 80 / 100):, ["grade","bathrooms","lat","sqft_living","view"]]

ytest = data.loc[(len(data) * 80 / 100):,["price"]]

# # add ones column
# Xtest.insert(0, 'Ones', 1)

# convert to matrices 
Xtrain = np.matrix(Xtrain.values)
ytrain = np.matrix(ytrain.values)

# Xtest = np.matrix(Xtest.values)
ytest = np.matrix(ytest.values)

# initialize theta
theta = np.matrix(np.array([0,0,0,0,0,0]))


#=========================================================================
# cost function
def costFunction(X, y, theta):
    z = np.power(((X * theta.T) - y), 2)
    return np.sum(z) / (2 * len(X))

# a)

# GD function
def gradientDescent(Xtest, ytest, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])   #1x2  shape[1] = 2 
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (Xtest * theta.T) - ytest
        
        for j in range(parameters):
            term = np.multiply(error, Xtest[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(Xtest)) * np.sum(term))
            
        theta = temp
        cost[i] = costFunction(Xtest, ytest, theta)
        
    return theta, cost


# initialize variables for learning rate and iterations
alpha = 0.1
iters = 500

# perform gradient descent to "fit" the model parameters

newTheta, cost = gradientDescent(Xtrain, ytrain, theta, alpha, iters)

print('newTheta = ' , newTheta)

# b) error on every iter
print('cost  = ' , cost)

trainingCost = costFunction(Xtrain, ytrain, newTheta)
print('training data cost = ' , trainingCost )
print('**************************************')


# c)

# h = newTheta * Xtest.T
# h = h.T
#print(h)


# h = c1 + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5

sqftLivingTest = data.loc[len(data) * 80 / 100:, ["sqft_living"]]
sqftLivingTest = np.matrix(sqftLivingTest.values)

gradeTest = data.loc[len(data) * 80 / 100:, ["grade"]]
gradeTest = np.matrix(gradeTest.values)

bathroomsTest = data.loc[len(data) * 80 / 100:, ["bathrooms"]]
bathroomsTest = np.matrix(bathroomsTest.values)


latTest = data.loc[len(data) * 80 / 100:, ["lat"]]
latTest = np.matrix(latTest.values)

viewTest = data.loc[len(data) * 80 / 100:, ["view"]]
viewTest = np.matrix(viewTest.values)


hypothesisFun = newTheta[0, 0] + (newTheta[0, 1] * gradeTest)+(newTheta[0, 2] * bathroomsTest) +(newTheta[0, 3] * latTest)+(newTheta[0, 4] * sqftLivingTest)+(newTheta[0, 5] * viewTest)


print("actual value: ", ytest)
print("predicted: ", hypothesisFun)


# d)

alpha = 0.005
theta_1, cost_history_1 = gradientDescent(Xtrain, ytrain, theta, alpha, iters)

alpha = 0.01
theta_2, cost_history_2 = gradientDescent(Xtrain, ytrain, theta, alpha, iters)

alpha = 0.02
theta_3, cost_history_3 = gradientDescent(Xtrain, ytrain, theta, alpha, iters)

alpha = 0.03
theta_4, cost_history_4 = gradientDescent(Xtrain, ytrain, theta, alpha, iters)

alpha = 0.15
theta_5, cost_history_5 = gradientDescent(Xtrain, ytrain, theta, alpha, iters)

plt.plot(range(1, iters +1), cost_history_1, color ='purple', label = 'alpha = 0.005')
plt.plot(range(1, iters +1), cost_history_2, color ='red', label = 'alpha = 0.01')
plt.plot(range(1, iters +1), cost_history_3, color ='green', label = 'alpha = 0.02')
plt.plot(range(1, iters +1), cost_history_4, color ='yellow', label = 'alpha = 0.03')
plt.plot(range(1, iters +1), cost_history_5, color ='blue', label = 'alpha = 0.15')

plt.rcParams["figure.figsize"] = (5,5)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("Effect of Learning Rate On Convergence of Gradient Descent")
plt.legend()






