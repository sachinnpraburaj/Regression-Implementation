#!/usr/bin/env python

import data_utils as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


train_err = dict()
test_err = dict()
for i in range(1,7):
    (w,train_err[i]) = a1.linear_regression(x_train,t_train,'polynomial',degree = i)
    (t_est,test_err[i]) = a1.evaluate_regression(x_test,t_test,w,'polynomial',degree = i)

# Produce a plot of results.
plt.plot(list(train_err.keys()), list(train_err.values()))
plt.plot(list(test_err.keys()), list(test_err.values()))
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
