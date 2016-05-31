#-*- encoding: utf-8 -*-
'''
Created on 2016-05-18

@author: zhangdan
'''
from __future__ import division
import csv
import numpy as np
import matplotlib.pyplot as plt


def get_training_data():
    
    temp_data = []
    with open("data\\hw2\\optdigits.tra") as csvfile:
        f_reader = csv.reader(csvfile)
        for line in f_reader:
            if int(line[64])==3:
                line.pop()
                temp_data.append(line) 
    data = np.zeros((len(temp_data), 64), dtype='int')
    for i in xrange(len(temp_data)):
        data[i] = np.array(temp_data[i], dtype='int')

    return  data

def pca(data, n_components):
    """
    data: MxN matrix of input data 
          (M-number of samples, N-data dimensions)
    n_components: Number of components to keep
    
    """
    #subtract off the mean for each dimension
    data -= np.mean(data, axis=0) 
    
    #calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=0)

    #find the eigenvectors and eigenvalues
    eig_vals, eig_vects = np.linalg.eig(cov_matrix)

    #sort the eig_vals in decreasing order
    eig_vals_sorted = np.argsort(eig_vals)[::-1]
    
    #choose the eigen vectors designated by the n_components
    eig_vals_inx = eig_vals_sorted[:n_components]
    eig_vects_inx = eig_vects[:, eig_vals_inx]
    
    #project the original data set
    data_pca = np.dot(data, eig_vects_inx)
    return eig_vects_inx, data_pca
    
    
    
    
    
if __name__ == '__main__':
    data = get_training_data()
    coff, data_pca = pca(data, 2)
    plt.plot(data_pca[:, 0], data_pca[:,1], 'or')
    plt.grid(True)
    plt.xlabel("first principal component")
    plt.ylabel("second principal component")
    plt.show()



      