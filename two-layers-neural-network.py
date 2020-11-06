# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:37:28 2020

@author: halil
"""
#%% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read_csv

data = pd.read_csv("data.csv")

data.drop(["Unnamed: 32","id"], axis = 1, inplace = True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis = 1)

#%% normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
# (x - min(x))/(max(x)-min(x))

#%% split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T



#%%create two layers neural network

#initialize parameters
def initialize_parameters(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters

#%%forward propagation
    
#sigmoid function
def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))
    return y_head

def forward_propagation(x_train, parameters):
    z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    a1 = np.tanh(z1)
    z2 = np.dot(parameters["weight2"], a1) + parameters["bias2"]
    a2 = sigmoid(z2)   
    cache = {"z1":z1,"a1":a1,"z2":z2,"a2":a2}   
    return cache

#%%compute cost
def compute_cost(a2, Y, parameters):
    logprobs = np.multiply(np.log(a2),Y)
    cost = -np.sum(logprobs)/Y.shape[0]
    return cost

#%%backward propagation
def backward_propagation(parameters, cache, X, Y):
    dz2 = cache["a2"]-Y
    dw2 = np.dot(dz2, cache["a1"].T)/X.shape[1]
    db2 = np.sum(dz2, axis = 1, keepdims = True)/X.shape[1]
    dz1 = np.dot(parameters["weight2"].T,dz2)*(1-np.power(cache["a1"],2))
    dw1 = np.dot(dz1, X.T)/X.shape[1]
    db1 = np.sum(dz1,axis=1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dw1,
             "dbias1":db1,
             "dweight2":dw2,
             "dbias2":db2}
    return grads

#%%update parameters
def update_parameters(parameters,grads,learning_rate = 0.01):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    return parameters
    
#%%prediction
def predict(parameters,x_test):
    cache = forward_propagation(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    for i in range(cache["a2"].shape[1]):
        if cache["a2"][0,i]<=0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    return Y_prediction


#%%Two Layer Neural Network
def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        cache = forward_propagation(x_train,parameters)
        # compute cost
        cost = compute_cost(cache["a2"], y_train, parameters)
         # backward propagation
        grads = backward_propagation(parameters, cache, x_train, y_train)
         # update parameters
        parameters = update_parameters(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters, cost_list, index_list

#%%
parameters, cost_list, index_list = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=10000)


#%%
# predict
y_prediction_test = predict(parameters,x_test)
y_prediction_train = predict(parameters,x_train)

# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

plt.plot(index_list,cost_list)
plt.xticks(index_list,rotation='vertical')
plt.xlabel("Number of Iterarion")
plt.ylabel("Cost")
plt.show()