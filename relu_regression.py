#!/usr/bin/env python

import data_utils as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10]

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

print ("Training Error:")
(w,train_err) = a1.linear_regression(x_train,t_train,'ReLU')
print (features[10],"------------",train_err,"\n")

print ("Test Error:")
(t_est,test_err) = a1.evaluate_regression(x_test,t_test,w,'ReLU', degree = 0)
print (features[10],"------------",test_err,"\n")

basis = 'ReLU'
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
(y_ev, test_err)  = a1.evaluate_regression(x_ev.reshape(500,1),x_ev.reshape(500,1),w,basis,degree = 0)


plt.plot(x_ev,y_ev,'r.-')
plt.plot(x_test,t_test,'g.')
plt.plot(x_train,t_train,'b.')
plt.title('A visualization of a regression estimate using random outputs')
plt.legend(['GNI','Training Points','Testing Points'])
plt.show()
