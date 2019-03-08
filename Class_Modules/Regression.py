#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



def OLS(Y, Y_hat):
    return np.sum((Y - Y_hat)**2)

def R2(y, y_hat):
    return 1 - (OLS(y, y_hat) / OLS(y, y.mean()))

def OLS_M(Y, Y_hat):
    return np.trace((Y - Y_hat).T.dot(Y - Y_hat))

def R2_M(Y, Y_hat):
    return 1 - ((Y - Y_hat)**2).sum(axis = 0) / ((Y - Y_hat.mean(axis = 0))**2).sum(axis = 0)



class Regression():
    
    def __init__(self):
        pass
    
    def Fit_GRB(self,x,y,eta=1e-6 ,epochs=1e4, lambda1 = 0, lambda2 = 0,batch_sz = 1,show_curve = False):
        N = x.shape[0]
        self.Y = y
        self.J = []
        
        PHI = np.ones((N,1))

        for i in range(N):
            PHI = np.column_stack((PHI, np.exp(-(x - x[i])**2)))
            
        self.X = PHI
    
        P = PHI.shape[1]
        self.w = np.random.randn(P) / np.sqrt(P)

        for i in range (epochs):
            
            for i in range (N//batch_sz):
                
                x_i = self.X[i*batch_sz:(i+1)*batch_sz,:]
                y_i = self.Y[i*batch_sz:(i+1)*batch_sz]
                
                y_hat = x_i.dot(self.w)
                self.w -= eta* (x_i.T.dot(y_hat - y_i) + lambda1*np.sign(self.w) + lambda2*self.w)
                
            self.y_hat = self.X.dot(self.w)
            self.J.append(OLS(self.Y,self.y_hat)+(lambda2/2)* np.sum(self.w.dot(self.w)) + lambda1 *np.sum(np.abs(self.w)))
   
        if (show_curve):
            plt.plot(self.J)
            
    def predict(self,x):
        N = x.shape[0]
        X = np.ones((N,1))
        for i in range(N):
            X = np.column_stack((X, np.exp(-(x - x[i])**2)))
        return X.dot(self.w)
    
    def Fit(self,x,y,eta=1e-6 ,epochs=1e4, lambda1 = 0, lambda2 = 0,batch_sz = 1,show_curve = False):
        N = x.shape[0]
        self.Y = y
        self.J = []
        self.X = np.vstack([np.array([1]*(N)), x.T]).T
        self.w = np.random.randn(self.X.shape[1])
        for i in range (epochs):
            
            for i in range (N//batch_sz):
                
                x_i = self.X[i*batch_sz:(i+1)*batch_sz,:]
                y_i = self.Y[i*batch_sz:(i+1)*batch_sz]
                
                y_hat = x_i.dot(self.w)
                self.w -= eta* (x_i.T.dot(y_hat - y_i) + lambda1*np.sign(self.w) + lambda2*self.w)
                
            self.y_hat = self.X.dot(self.w)
            self.J.append(OLS(self.Y,self.y_hat)+(lambda2/2)* np.sum(self.w.dot(self.w)) + lambda1 *np.sum(np.abs(self.w)))
   
        if (show_curve):
            plt.plot(self.J)
            
            
    def Fit2(self,x,y,eta=1e-6 ,epochs=1e4, lambda1 = 0, lambda2 = 0,batch_sz = 1,show_curve = False):
        N = x.shape[0]
        self.Y = y
        self.J = []
        self.X = np.vstack([np.array([1]*(N)), x.T]).T
        self.w = np.random.randn(self.X.shape[1],self.Y.shape[1])
        for i in range (epochs):
            
            for i in range (N//batch_sz):
                
                x_i = self.X[i*batch_sz:(i+1)*batch_sz,:]
                y_i = self.Y[i*batch_sz:(i+1)*batch_sz,:]
                
                y_hat = x_i.dot(self.w)
                self.w -= eta* (x_i.T.dot(y_hat - y_i) + lambda1*np.sign(self.w) + lambda2*self.w)
                
            self.y_hat = self.X.dot(self.w)
            
            self.J.append(OLS_M(self.Y,self.y_hat)+ ((lambda2/2)* np.sum(self.w*self.w)) + lambda1 *np.sum(np.abs(self.w)))
   
        if (show_curve):
            plt.plot(self.J)
            
    def predict(self,x):
        N = x.shape[0]
        X = np.ones((N,1))
        for i in range(N):
            X = np.column_stack((X, np.exp(-(x - x[i])**2)))
        return X.dot(self.w)
            
    def R2(self):
        return 1 - OLS(self.Y, self.y_hat) / OLS(self.Y,self.Y.mean())
    
    

