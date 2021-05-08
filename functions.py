# -*- coding: utf-8 -*-
"""
author: anonymous

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
        print("not implemented")
    

class ackley:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-32.768,32.768)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='ackley'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(1)/self.input_dim))-np.exp(np.cos(2*np.pi*X).sum(1)/self.input_dim))

        return self.ismax*fval


class Levy(functions):
    '''
    Egg holder function
    '''
    def __init__(self, input_dim, bounds=None, sd=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-10, 10)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(1.)*self.input_dim]
        self.fmin = 0
        self.ismax = -1
        self.name = 'Levy'

    def func(self, X):
        X = reshape(X, self.input_dim)

        w = np.zeros((X.shape[0], self.input_dim))
        for i in range(1, self.input_dim+1):
            w[:, i-1] = 1 + 1/4*(X[:, i-1]-1)

        fval = (np.sin(np.pi*w[:, 0]))**2 + ((w[:, self.input_dim-1]-1)**2)*(1+(np.sin(2*np.pi*w[:, self.input_dim-1]))**2)
        for i in range(1, self.input_dim):
            fval += ((w[:, i]-1)**2)*(1+10*(np.sin(np.pi*w[:, i]))**2) 

        return self.ismax*fval


class rosenbrock(functions):
    '''
    rosenbrock function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-2.048, 2.048)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax = -1
        self.name = 'Rosenbrock'
    
    def func(self, X):
        X = reshape(X, self.input_dim)
        fval = 0
        for i in range(self.input_dim-1):
            fval += (100*(X[:, i+1]-X[:, i]**2)**2 + (X[:, i]-1)**2)
        
        return self.ismax*fval


class alpine:
    '''
    Alpine1 function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None, sd=None):
        if bounds is None:
            self.bounds = [(0, 10)]*input_dim
        else:
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim
        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

        self.ismax = -1
        self.name = 'alpine1'

    def func(self, X):
        X = reshape(X, self.input_dim)
        temp = abs(X*np.sin(X) + 0.1*X)
        if len(temp.shape) <= 1:
            fval = np.sum(temp)
        else:
            fval = np.sum(temp, axis=1)

        return self.ismax*fval
