#!/usr/bin/env python

import data_utils as a1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
reg_constant = [0, .01, .1, 1, 10, 10**2 , 10**3 , 10**4]

train_err_reg = []
test_err_reg = []

kf = KFold(n_splits = 10)

for i in range(8):
    tot_test_err = []
    for train_index, test_index in kf.split(x_train):
        xc_train, xc_test = x_train[train_index], x_train[test_index]
        tc_train, tc_test = t_train[train_index], t_train[test_index]
        (w, train_err) = a1.linear_regression(xc_train,tc_train,'polynomial',reg_constant[i],2)
        (t_est, test_err) = a1.evaluate_regression(xc_test,tc_test,w,'polynomial',2)
        tot_test_err.append(test_err)

    test_err_reg.append(np.mean(tot_test_err))

print("Best lambda : ",reg_constant[test_err_reg.index(min(test_err_reg))])
print("Test error for best lambda : ", min(test_err_reg))
plt.semilogx(reg_constant,test_err_reg)
plt.show()
