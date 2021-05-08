# -*- coding: utf-8 -*-
"""
author: Huong Ha

"""

import numpy as np


def reshape(x, input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x


class functions:
    def plot(self):
        print("Not implemented")


class Protein_wl_Test:
    def __init__(self, input_dim, bounds=None, sd=None, seed=0):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax = 1
        self.name = 'Material_Fcc_Test'
        self.seed = seed
    
    def func(self, xx, X_data, Y_data):
        '''
        Call Protein_wl
        '''
        # Compute some parameters
        X_test = np.atleast_2d(xx)
        Y_test = []
        for i in range(X_test.shape[0]):
            Y_test_t = Y_data[np.where(np.all(X_data[:, :]==X_test[i, :],axis=1))[0][0], 0]
            Y_test.append(Y_test_t)
        Y_test = np.expand_dims(np.array(Y_test), axis=1)

        return Y_test


class DeepPerf_Assurance_Test:
    def __init__(self, input_dim, bounds=None, sd=None, seed=0):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax = 1
        self.name = 'DeepPerf_Assurance_Test'
        self.seed = seed
    
    def func(self, xx, model, max_X, max_Y, whole_data):
        '''
        Call DeepPerf
        '''
        
        # Compute some parameters
        (N, n) = whole_data.shape
        n = n-1
        X_test = np.atleast_2d(xx)
        Y_test = []
        for i in range(X_test.shape[0]):
            Y_test_t = whole_data[np.where(np.all(whole_data[:, 0:n]==X_test[i, :],axis=1))[0][0], n]
            Y_test.append(Y_test_t)
        Y_test = np.expand_dims(np.array(Y_test), axis=1)
        
        # Build and train model now 
        X_test = np.divide(X_test, max_X)
        
        Y_pred_test = model.predict(X_test)
        Y_pred_test = max_Y*Y_pred_test
        abs_error = np.abs(Y_test.ravel() - Y_pred_test.ravel())

        return abs_error
