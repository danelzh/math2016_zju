#-*- encoding: utf-8 -*-
'''
Created on 2016-05-20

@author: zhangdan
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate samples form multivariate normal distribution with 'numpy' package
def gen_mulgaussian_samples(mean_vec, cov_matrix, sample_size):
    return  np.random.multivariate_normal(mean_vec, cov_matrix, sample_size)

def model_probability(x, mu, sig):
    n = x.shape[1]
    meanDiff = x - mu
    pdf = 1 / np.sqrt((2 * np.pi) ** n * np.linalg.det(sig)) * np.exp(
          -1/2 * np.sum(np.dot(meanDiff, np.linalg.inv(sig)) * meanDiff, axis=1))
    return pdf
    
def init_parameters(sample):
    num = sample.shape[0]
    k = 2 # the number of clusters
          
    # Randomly select k data points to serve as the initial means.
    indices = np.random.permutation(num)
    mu = sample[indices[0:k], :]
    
    # Use the overall covariance of the dataset as the initial variance for each cluster
    sig = []
    for ii in xrange(k):
        sig.append(np.cov(sample, rowvar=0))
    
    # Assign equal prior probabilities to each cluster.
    phi = np.ones((k, )) * (1 / k)
    
    return num, k, mu, sig, phi

def mog_em(sample, cluster_num, mu, sig, phi):
    
    # Matrix to hold the probability that each data point belongs to each cluster.
    # One row per data point, one column per cluster.
    sample_num, sample_dim = sample.shape
    for iter in xrange(10000):
        print "EM iteration: %d\n" % iter
        # Expectation
        # Calculate the probability for each data point for each distribution.
        # Matrix to hold the pdf value for each every data point for every cluster.
        # One row per data point, one column per cluster.
        pdf_value = np.zeros((sample_num, cluster_num))
        for cl in xrange(cluster_num):
            #pdf_value[:, cl] = multivariate_normal.pdf(sample, mu[cl, :], sig[cl])
            pdf_value[:, cl] = model_probability(sample, mu[cl, :], sig[cl])
        pdf_phi = pdf_value * phi
        pdf_weight = pdf_phi / np.sum(pdf_phi, axis=1)[:, np.newaxis]
        
        # Maximization
        # Store the previous means
        prev_mu = mu.copy()
        for cl in xrange(cluster_num):
            phi[cl] = np.mean(pdf_weight[:, cl])
            val = np.dot(pdf_weight[:, cl].T, sample)
            mu[cl, :] = val / np.sum(val)
            
            sig_k = np.zeros((sample_dim, sample_dim))
            sample_mean_removed = sample - mu[cl, :]

            for num in xrange(sample_num):
                sig_k = sig_k + pdf_weight[num, cl] * np.dot(sample_mean_removed[num, :][np.newaxis, :].T, 
                                                             sample_mean_removed[num, :][np.newaxis, :])
            
            sig[cl] = sig_k / np.sum(pdf_weight[:, cl])

        if np.sum(np.abs(prev_mu - mu)) <=  1e-15:
            break
    
    return pdf_value, mu, sig
        




if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

    mu1, mu2 = [1, 2], [-1, -2]
    sig1, sig2 = [[3, 0], [0, 2]], [[2, 0], [0, 1]]
    num1, num2 = 300, 200
    sample1 = gen_mulgaussian_samples(mu1, sig1, num1)
    sample2 = gen_mulgaussian_samples(mu2, sig2, num2)
    sample = np.vstack((sample1, sample2))
    
    
    # Plot the data points 
    plt.figure(0)
    plt.plot(sample1[:, 0], sample1[:, 1], 'bo')
    plt.plot(sample2[:, 0], sample2[:, 1], 'ro')
    plt.title('Original data')
    # plt.show()
    
    # Choose initial values for the parameters
    num, k, mu, sig, phi = init_parameters(sample)

    # Run EM algorithm
    pr, mu_est, sig_est = mog_em(sample, k, mu, sig, phi)
    labels = np.argmax(pr, axis=1)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])   

    plt.figure(1)
    data_colors=[colors[lbl] for lbl in labels]
    plt.scatter(sample[:, 0], sample[:, 1], c=data_colors, alpha=0.5)
    plt.show()

    #y = pdf_model(None, None)
    #plt.plot(sample[:,0], sample[:, 1], 'xg')
    #plt.show()
    