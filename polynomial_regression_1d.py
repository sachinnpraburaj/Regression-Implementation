#!/usr/bin/env python

import data_utils as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:15]

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


train_err = dict()
test_err = dict()

for i in range(8):
    (w,train_err[i+8]) = a1.linear_regression(x_train[:,i],t_train,'polynomial',degree = 3)
    (t_est,test_err[i+8]) = a1.evaluate_regression(x_test[:,i],t_test,w,'polynomial',degree = 3)


# Produce a plot of results.
index = np.arange(8)
bar_width = 0.25

plt.bar(index, test_err.values(), bar_width,color='orange',label='Test Error')
plt.bar(index + bar_width + 0.1, train_err.values(), bar_width,color='green',label='Training Error')
plt.legend()

plt.ylabel('RMS')
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Feature')
plt.xticks(index + bar_width, train_err.keys())
plt.tight_layout()
plt.show()
