"""
Weighted and unweighted linear regression
"""

import sys
import numpy as np
from sklearn import linear_model, feature_selection, cross_validation, cluster
import matplotlib.pyplot as plt

from utils import reader, mappings, extractor

# Debugging: Controls how many lines reader reads in
LIMIT = None

YEAR = '2010'

class LinearRegressionModel(object):

	def __init__(self, X=None, Y=None):
		self.X = X
		if self.X != None:
			# Add intercept term to X
			self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
		self.Y = Y
		if self.X != None and self.Y != None:
			self._calcUnweightedTheta()

	def _calcUnweightedTheta(self):
		"""
		Calcualte unweighted theta using normal equation
		"""
		self.theta = np.dot(np.dot(
			np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.Y)

	def train(self, X, Y):
		# Add intercept term to X
		self.X = np.hstack((np.ones((X.shape[0], 1)), X))
		self.Y = Y
		self._calcUnweightedTheta()

	def h(self, x, weighted=True, tau=0.1):
		"""
		Make prediction on target value based on feature values x and
		current model
		tau is bandwidth parameter for weighted predictions
		"""
		# Add intercept term to x
		x = np.append([1], x)

		if weighted:
			# Calculate weight matrix
			m, k = self.X.shape
			W = np.zeros((m, m))
			for i in range(m):
				diff = self.X[i] - x
				W[i][i] = np.exp(0 - np.dot(diff, diff) / (2 * np.square(tau)))

			# Calculate weighted theta using normal equation
			theta = np.dot(np.dot(np.dot(
				np.linalg.inv(np.dot(np.dot(self.X.T, W), self.X)),
				self.X.T), W), self.Y)

			return np.dot(theta, x)

		else:
			return np.dot(self.theta, x)

	def predict(self, X, Y, weighted=False, tau=1):
		"""
		Make predictions based on X and compare to Y.
		Return the average error.
		"""
		error = float(0)
		for x, y in zip(X, Y):
			error += abs(self.h(x, weighted=weighted, tau=tau) - y)
		return error / len(Y)

	def trainingError(self, weighted=False, tau=1):
		"""
		Calculate average training error of hypothesis.
		"""
		return self.predict(self.X[:, 1:], self.Y, weighted=weighted, tau=tau)

featureCandidates = [(name, spec)
	for name, spec in mappings.features[YEAR]
	# if name in ['sex', 'race', 'height', 'weight']]
	if spec[2] == int or spec[2] == float or spec[2] == bool]

def extractFeatures(line):
	res = []
	for (name, spec) in featureCandidates:
		value = extractor.extract(line, name, spec)
		res.append(value)
		# if name in ['age', 'height', 'weight', 'temperature']:
		# 	res.append(pow(value, 3))
		# 	res.append(pow(value, 2))
		# 	if value > -7:
		# 		res.append(pow(value, .5))
		# 		res.append(pow(value, 1/3))
		# 	else:
		# 		res.append(0)
		# 		res.append(0)
	return res

def extractTarget(line):
	# Extract field rounded down to nearest bucket size
	return extractor.extract(line, 'timeWithMD', mappings.target[YEAR][1])

def cv(model, X, Y):
	# Cross validation
	testErrors = []
	trainingErrors = []
	predictions = []

	kf = cross_validation.KFold(len(Y), n_folds=10)
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		model.fit(X_train, Y_train)
		for i in range(len(Y_train)):
			prediction = model.predict(X_train[i])
			trainingErrors.append(np.absolute(prediction - Y_train[i]))
			predictions.append(prediction)
		# print 'training error: ', np.mean(np.absolute(model.predict(X_train) - Y_train))
		for i in range(len(Y_test)):
			prediction = model.predict(X_test[i])
			testErrors.append(np.absolute(prediction - Y_test[i]))
			predictions.append(prediction)
		# print 'test error: ', np.mean(np.absolute(model.predict(X_test) - Y_test))

	# print 'avg training error: ', np.mean(trainingErrors)
	# print 'avg test error: ', np.mean(testErrors)
	return (np.mean(trainingErrors), np.mean(testErrors))

	# plt.figure()
	# plt.hist(testErrors, 30)
	# plt.show()
	# plt.figure()
	# plt.hist(trainingErrors, 30)
	# plt.show()
	# plt.figure()
	# plt.hist(predictions)
	# plt.show()

def fsel(model, X, Y):
	# Feature selection
	selector = feature_selection.RFECV(model)
	selector = selector.fit(X, Y)
	# print selector.support_
	# print selector.ranking_
	# print selector.grid_scores_
	selectedFeatures = []
	for i in range(len(selector.support_)):
		if selector.support_[i]:
			if i < len(featureCandidates):
				selectedFeatures.append(featureCandidates[i][0])
			else:
				selectedFeatures.append('km-%s' % (i - len(featureCandidates)))
	print 'selected features:', selectedFeatures
	X = selector.transform(X)
	return X

def addKMeansFeatures(X, Y):
	K = 50
	print 'added k-means features, k =', K
	km = cluster.KMeans(n_clusters=K)
	return np.hstack((X, km.fit_transform(X, Y)))

def plotTrainingTestError(model, X, Y):
	ms = [20, 50, 100] + range(200, len(Y), len(Y) / 25)
	trainErrors = []
	testErrors = []
	for m in ms:
		trainError, testError = cv(model, X[:m], Y[:m])
		trainErrors.append(trainError)
		testErrors.append(testError)
	plt.plot(ms, trainErrors, 'g-')
	plt.plot(ms, testErrors, 'r-')
	print ms
	print trainErrors
	print testErrors
	plt.show()

def main(argv):
	if len(argv) < 2:
		print "Usage: python linear_regression.py <data>"
		sys.exit(1)

	Y, X = reader.read(argv[1], **{
		'extractFeaturesFn': extractFeatures,
		'extractLabelsFn': extractTarget,
		'limit': LIMIT
	})

	# Take out invalid values
	takeOutInvalid = False
	XY = np.array([xy for xy in np.hstack((X, Y.reshape(-1, 1)))
			if not takeOutInvalid or all([i > -7 for i in xy])])
	XY = np.random.permutation(XY)
	X = XY[:, :-1]
	Y = XY[:, -1]
	print len(Y)

	# Applying average feature values to invalid values
	averages = []
	numInvalidByFeature = []
	for j in range(len(X[0])):
		averages.append(sum([x[j] for x in X]) / len(X))
		numInvalidByFeature.append(0)
	numInvalid = 0
	numBadX = 0
	numInvalidInX = 0
	cleanX = []
	cleanY = []
	for i in range(len(X)):
		numInvalidInX = 0
		for j in range(len(X[0])):
			if X[i][j] <= -7:
				X[i][j] = averages[j]
				numInvalid = numInvalid + 1
				numInvalidInX = numInvalidInX + 1
				numInvalidByFeature[j] = numInvalidByFeature[j] + 1
		if numInvalidInX > len(X[0]) / 15:
			numBadX = numBadX + 1
		else:
			cleanX.append(X[i])
			cleanY.append(Y[i])
	print numInvalid
	# 107123
	print numBadX
	# 9854 xs have >1/10 invalid cells
	# 14582 xs have >1/15 invalid cells
	# 19452 xs have >1/20 invalid cells

	# Take out invalid features (i.e. features with too many invalid values)
	invalidFeatures = [] # Indices of invalid features
	for j in range(len(X[0])):
		if numInvalidByFeature[j] > len(X) / 10:
			invalidFeatures.append(j)
	for i in range(len(cleanX)):
		cleanX[i] = np.delete(cleanX[i], invalidFeatures)

	X = np.array(cleanX)
	Y = np.array(cleanY)
	print X.shape, Y.shape

	# Add K-means features
	X = addKMeansFeatures(X, Y)

	# Create model

	# model = linear_model.LinearRegression()

	# model = linear_model.Lasso(alpha=.01)

	# model = linear_model.LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
	# model.fit(X, Y)
	# print model.alpha_

	# model = linear_model.Ridge(alpha=100)

	model = linear_model.RidgeCV(normalize=True, alphas=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 10, 100, 1000, 10000, 100000])
	model.fit(X, Y)
	print 'alpha:', model.alpha_

	# model = linear_model.ElasticNet(alpha=.1, l1_ratio=.1)

	model.fit(X, Y)
	print 'intercept:', model.intercept_
	print 'coef:', model.coef_

	# Feature selection, increases performance a lot
	X = fsel(model, X, Y)

	# Error over m
	plotTrainingTestError(model, X, Y)

	# print cv(model, X, Y)


	# OLD CODE
	# model = LinearRegressionModel()

	# Run cross validation
	# print crossValidate(model, X, Y, cvMethod='kfold', weighted=True, tau=1)

	# Train on one feature at a time
	# for f in range(X.shape[1]):
	# 	xx = np.take(X, [f], axis=1)
	# 	xx = np.array([x for x in xx if all([i > -7 for i in x])])
	# 	try:
	# 		model = LinearRegressionModel()
	# 		e = crossValidate(model, xx, Y, cvMethod='simple', weighted=True)
	# 		print featureCandidates[f][0], model.theta, e
	# 	except Exception:
	# 		print 'Exception'
	# 		continue

	# Run forward search
	# features, testError, trainingError = forwardSearch(
	# 	model, X, Y, cvMethod='simple', weighted=False)
	# print len(featureCandidates), features
	# print [featureCandidates[i][0] for i in features]
	# print testError, trainingError

	# Error-m relation
	# ms = [30, 300, 3000, 30000]
	# result = []
	# for m in ms:
	# 	_, testError, trainingError = forwardSearch(model, X[:m], Y[:m], cvMethod='simple')
	# 	result.append((m, testError, trainingError))
	# 	print m, testError, trainingError
	# print result

	# Train and test with imaginary dataset
	# m = 10
	# X = np.array([[i] for i in range(m)])
	# Y = np.array([i * i for i in range(m)])
	# model.train(X, Y)
	# print model.h(np.array([5]))
	# print model.trainingError(weighted=True, tau=0.5)
	# print crossValidate(model, X, Y, weighted=True)


if __name__ == '__main__':
	main(sys.argv)
