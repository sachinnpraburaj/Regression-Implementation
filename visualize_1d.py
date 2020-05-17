#!/usr/bin/env python

import data_utils as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
# x = a1.normalize_data(x)
col = 3
N_TRAIN = 100;
# Select a single feature.
x_train = x[0:N_TRAIN,col]
x_test = x[N_TRAIN:,col]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

basis = 'polynomial'
# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
(w,train_err) = a1.linear_regression(x_train,t_train,basis,degree = 3)
(y_ev, test_err)  = a1.evaluate_regression(x_ev.reshape(500,1),x_ev.reshape(500,1),w,basis,degree = 3)


plt.plot(x_ev,y_ev,'r.-')
plt.plot(x_test,t_test,'g.')
plt.plot(x_train,t_train,'b.')
plt.title('A visualization of a regression estimate using random outputs')
plt.legend(['GNI','Training Points','Test Points'])
plt.show()
