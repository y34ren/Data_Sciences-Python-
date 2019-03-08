#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:
def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def softmax(H):
	eH = np.exp(H)
	return eH/eH.sum(axis = 1, keepdims = True)

def accuracy(y, y_hat):
	return np.mean(y == y_hat)

def cross_entropy(Y, P_hat):
	return -np.sum(Y*np.log(P_hat))

def one_hot_encode(y):
	N = len(y)
	K = len(set(y))

	Y = np.zeros((N,K))

	for i in range(N):
		Y[i,y[i]] = 1

	return Y


class Classification():
    
    def __init__(self):
        pass
        
    def Fit(self,X,Y, eta= 1e-8 ,epochs= int(1e3), lambda1 = 0, lambda2 = 0, batch_sz = 1,show_curve = False):
        
        self.X = np.hstack((np.ones((X.shape[0],1)), X))
        if (len(np.unique(Y))>2):
            self.Binomial = False
        else:
            self.Binomial = True
        
        if (self.Binomial):
            self.Y = Y
            self.F = sigmoid
            self.w = np.random.randn(self.X.shape[1])
        else:
            self.Y = one_hot_encode(Y)
            self.F = softmax
            self.w = np.random.randn(self.X.shape[1],self.Y.shape[1])
          
        self.J = []

        for i in range(epochs):
            
            for i in range(self.X.shape[0]//batch_sz):
                x_i = self.X[i*batch_sz:(i+1)*batch_sz,:]
                if self.Binomial:
                    y_i = self.Y[i*batch_sz:(i+1)*batch_sz]
                else:
                    y_i = self.Y[i*batch_sz:(i+1)*batch_sz,:]
                
                p = self.F(x_i.dot(self.w)) 
                self.w -= eta*(x_i.T.dot(p - y_i) + lambda1*np.sign(self.w) + lambda2*self.w)
   

            self.p = self.F(self.X.dot(self.w))
            self.J.append((cross_entropy(self.Y, self.p)) + (lambda2/2)* np.sum(self.w.dot(self.w)) + lambda1 *np.sum(np.abs(self.w)))
                    
        if (show_curve):
            plt.plot(self.J)
            
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        p = self.F(X.dot(self.w))
        if self.Binomial:
            return np.round(p)
        else:
            return p.argmax(axis = 1)
            
    def accuracy (self):
        if self.Binomial:
            return np.mean(self.Y == np.round(self.p))
        else:
            return np.mean(self.Y.argmax(axis=1) == self.p.argmax(axis =1))
    
    


# In[ ]:




