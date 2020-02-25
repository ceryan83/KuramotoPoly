#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 08:56:01 2019

@author: eabernal
"""
#%% dependencies and definitions
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    
    def __init__(self, degree = 1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(degree*(degree+3)//2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 20)
        self.fc6 = nn.Linear(20, 20)
        self.fc7 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return F.log_softmax(x)

#%% plot function
   
def plot_decision_boundary(clf, X, y, filename):
    # Set min and max values and give it some padding
    #x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    #y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    X_out = net(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype = torch.float))
    Z = X_out.data.max(1)[1]
    # Z.shape
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.savefig(filename)
    plt.close()

#%% read and process data

data = pd.read_csv('NearBoundary+NearCenter_kuramoto3')
X = data.values[:, 0:2]  # Take only the first two features.
X = torch.tensor(X, dtype = torch.float)
y = data.values[:, 2]
y = 1*(y<=2)
y = torch.tensor(y, dtype = torch.long)
for n in range(1,6):
    Xpoly = X
    for k in range(2,n+1):
        for i in range(k+1):
            j = k-i
            xpoly = torch.mul(torch.pow(X[:,0],i),torch.pow(X[:,1],j)) #creates polynomial features
            xpoly = torch.reshape(xpoly, (torch.numel(xpoly),1))
            Xpoly = torch.cat((Xpoly,xpoly), 1) #joins polynomial features to data matrix


#%% train
    net = Net(n)
    
    # create a stochastic gradient descent optimizer
    learning_rate = 0.3
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.CrossEntropyLoss()
    
    nepochs = 1000
    data, target = Xpoly, y
    # run the main training loop
    for epoch in range(nepochs):
        if epoch % 3000 == 0 and epoch <= 24000:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']/2
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('Epoch ', epoch, 'Loss ', loss.item())
        
#%% compute accuracy on training data
    
    net_out = net(data)
    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    correctidx = pred.eq(target.data) 
    ncorrect = correctidx.sum()
    accuracy = ncorrect.item()/len(data)
    print('Training accuracy of degree %d is ' %(n), accuracy)

#%% if need to train further

#for epoch in range(nepochs):
#    # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
#    optimizer.zero_grad()
#    net_out = net(data)
#    loss = criterion(net_out, target)
#    loss.backward()
#    optimizer.step()
#    if epoch % 100 == 0:
#        print('Epoch ', epoch, 'Loss ', loss.item())


#%%  plot outputs
"""
plot_decision_boundary(net, X, y, 'Results_7layer_9741.pdf')
plot_decision_boundary(net, X[correctidx,:], y[correctidx], 'Correct_7layer_9741.pdf')
plot_decision_boundary(net, X[~correctidx,:], y[~correctidx], 'Inorrect_7layer_2.pdf')
"""

#%% save model
"""
torch.save(net, './Model6L0.3LR50kepoch')
"""
#%% test model on other data
#net1 = torch.load('./Model6L0.3LR150kepoch')
#net1 = torch.load('./Model6L0.3LR150kepoch9741')
"""
data = pd.read_csv('Uniform_kuramoto3')
X_test = data.values[:, 0:2]  # Take only the first two features.   
X_test = torch.tensor(X_test, dtype = torch.float)
Xpoly_test = X_test
if n >=2:
    for k in range(2,n+1):
        for i in range(k+1):
            j = k-i
            xpoly = torch.mul(torch.pow(X_test[:,0],i),torch.pow(X_test[:,1],j)) #creates polynomial features
            xpoly = torch.reshape(xpoly, (torch.numel(xpoly),1))
            Xpoly_test = torch.cat((Xpoly_test,xpoly), 1) #joins polynomial features to data matrix   
y_test = data.values[:, 2]
y_test = 1*(y<=2)
y_test = torch.tensor(y, dtype = torch.long)
net_out = net(Xpoly_test)
pred = net_out.data.max(1)[1]
correctidx = pred.eq(y_test.data) 
ncorrect = correctidx.sum()
accuracy = ncorrect.item()/len(data)
print('Training accuracy is ', accuracy)

incorrectidx = ~pred.eq(y_test.data) 
torch.nonzero(incorrectidx)
X_test[torch.nonzero(incorrectidx),:]
torch.nonzero(correctidx)

data = pd.read_csv('NearBoundary_kuramoto3')
X_test = data.values[:, 0:2]  # Take only the first two features.   
X_test = torch.tensor(X_test, dtype = torch.float)
Xpoly_test = X_test
if n >=2:
    for k in range(2,n+1):
        for i in range(k+1):
            j = k-i
            xpoly = torch.mul(torch.pow(X_test[:,0],i),torch.pow(X_test[:,1],j)) #creates polynomial features
            xpoly = torch.reshape(xpoly, (torch.numel(xpoly),1))
            Xpoly_test = torch.cat((Xpoly_test,xpoly), 1) #joins polynomial features to data matrix   
y_test = data.values[:, 2]
y_test = 1*(y<=2)
y_test = torch.tensor(y, dtype = torch.long)
net_out = net(Xpoly_test)
pred = net_out.data.max(1)[1]
correctidx = pred.eq(y_test.data) 
ncorrect = correctidx.sum()
accuracy = ncorrect.item()/len(data)
print('Test accuracy is ', accuracy)
"""
