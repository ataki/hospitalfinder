"""
Cross validator

Usage:

	from cv import crossValidate
	error = crossValidate(mlModel, X, Y, cvMethod='simple'|'kfold', **options)
	# options is passed to mlModel.predict.

"""

import numpy as np

def crossValidate(model, X, Y, cvMethod='simple', kfold_level=10, **options):
	"""
	Run cross validation on a model using training samples X, Y
	Pass options parameter to model.predict method.
	"""
	# Randomly permutate data
	permutated = np.random.permutation(np.hstack((X, Y.reshape(-1, 1))))
	X = permutated[:, :-1]
	Y = permutated[:, -1]

	if cvMethod == 'simple':
		trainX, trainY, testX, testY = _getSimpleCVData(X, Y)
		model.train(trainX, trainY)
		return model.predict(testX, testY, **options)

	elif cvMethod == 'kfold':
		error = float(0)
		for i in range(kfold_level):
			trainX, trainY, testX, testY = _getKFoldCVData(X, Y, i, kfold_level)
			model.train(trainX, trainY)
			error += model.predict(testX, testY, **options)
		return error / kfold_level

def _getSimpleCVData(X, Y):
	"""
	Splits dataset into 70% / 30%.
	Returns (trainX, trainY, testX, testY).
	"""
	split = int(len(Y) * 0.7)
	return (X[:split], Y[:split], X[split:], Y[split:])

def _getKFoldCVData(X, Y, i, kfold_level):
	"""
	Get training and testing data for the ith iteration of K-fold CV.
	Returns (trainX, trainY, testX, testY).
	"""
	start = int(i * len(Y) / kfold_level) # Start index of test data
	end = int((i + 1) * len(Y) / kfold_level) # End index of test data

	return (np.vstack((X[:start], X[end:])), np.hstack((Y[:start], Y[end:])),
			X[start:end], Y[start:end])
