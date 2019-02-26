import numpy as np


def one_hot_encode(y):
	N = len(y)
	K = len(set(y))

	Y = np.zeros((N,K))

	for i in range(N):
		Y[i,y[i]] = 1

	return Y


def shuffle(*args):
	idx = np.random.permutation(len(args[0]))
	return [X[idx] for X in args]
