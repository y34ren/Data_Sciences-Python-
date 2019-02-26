import numpy as np


def OLS(Y, Y_hat):
	return np.sum((Y - Y_hat)**2)


def cross_entropy(Y, P_hat):
	return -np.sum(Y*np.log(P_hat))
