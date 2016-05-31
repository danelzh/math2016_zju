#-*- encoding: utf-8 -*-
'''
Created on 2016-05-18

@author: zhangdan
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

sample_num = 10     
polynomial_order = 9
l_mu = 0
l_sigma = 0.1
l_lambda = np.exp(-18)

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

#the true curve of sin
x_sin = np.linspace(0, 1, sample_num * 10)
y_sin = np.sin(2*np.pi*x_sin)
plt.plot(x_sin, y_sin, 'g')


#samples with Gaussian noise
x_noisy = np.linspace(0, 1, sample_num)
y_noisy = np.sin(2*np.pi*x_noisy) + np.random.normal(l_mu, l_sigma, sample_num)
plt.plot(x_noisy, y_noisy, 'ko', fillstyle='none')

#fit a 2-D matrix with the data samples
data = np.zeros((sample_num, polynomial_order+1))
for i in xrange(sample_num):
    for j in xrange(polynomial_order+1):
        data[i, j] = np.power(x_noisy[i], j)

#weight = np.zeros((sample_num, 1))
label = np.array(y_noisy)[:, np.newaxis]

#calculate the weight without regularization
weight = np.dot (np.dot(np.linalg.inv(np.dot(data.T, data)), data.T), label)

#calculate the weight with regularization
weight_reg = np.dot (np.dot(np.linalg.inv((np.dot(data.T, data)) + l_lambda * np.identity(polynomial_order+1)), 
                            data.transpose()), label)   


#generate points to plot curve with weight and weight_reg respectively
fitting_x = np.zeros((sample_num * 5, polynomial_order+1))
plotx = np.linspace(0, 1, sample_num * 5)
for i in xrange(sample_num  * 5):
    for j in xrange(polynomial_order+1):
        fitting_x[i, j] = np.power(plotx[i], j)

fitting_y = np.dot(fitting_x, weight)
fitting_y_reg = np.dot(fitting_x, weight_reg)

#plot the fitting curve
plt.plot(plotx, fitting_y[:, 0], '--r')
plt.plot(plotx, fitting_y_reg[:, 0], '--r')

plt.xlabel('x')
plt.ylabel('t')
 
plt.xlim(-0.1, 1.1) 
plt.ylim(-1.2, 1.2)
 
plt.legend(['True', 'Noisy', 'Fitting', 'Regularized'])
#plt.legend(['True', 'Noisy Samples', 'Regularized'])
plt.show()

