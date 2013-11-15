"""
Weighted and unweighted linear regression
"""

from utils import reader, mappings, extractor
import sys
import numpy as np
from utils.cv import crossValidate
from utils.fsel import forwardSearch

# Debugging: Controls how many lines reader reads in
LIMIT = 30

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

	def h(self, x, weighted=True, tau=1):
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

	def predict(self, X, Y, weighted=False, tau=3):
		"""
		Make predictions based on X and compare to Y.
		Return the average error.
		"""
		error = float(0)
		for x, y in zip(X, Y):
			error += abs(self.h(x, weighted=weighted, tau=tau) - y)
		return error / len(Y)

	def trainingError(self, weighted=False, tau=3):
		"""
		Calculate average training error of hypothesis.
		"""
		return self.predict(self.X[:, 1:], self.Y, weighted=weighted, tau=tau)

featureCandidates = [(name, spec)
	for name, spec in mappings.features['2010']
	if spec[2] == int or spec[2] == float or spec[2] == bool]

def extractFeatures2010(line):
	"""
	Extract features based on specs from 2010
	"""
	return [extractor.extract(line, spec) for (name, spec) in featureCandidates]

def extractTarget(line):
	# Extract field rounded down to nearest bucket size
	return extractor.extract(line, mappings.target['2010'][1])

def main(argv):
	if len(argv) < 2:
		print "Usage: python linear_regression.py <data>"
		sys.exit(1)

	Y, X = reader.read(argv[1], **{
		'extractFeaturesFn': extractFeatures2010,
		'extractLabelsFn': extractTarget,
		'limit': LIMIT
	})

	model = LinearRegressionModel()

	# Run cross validation
	# print crossValidate(model, X, Y, cvMethod='kfold', weighted=True, tau=5)

	# Train on one feature at a time
	# for f in range(X.shape[1]):
	# 	xx = np.take(X, [f], axis=1)
	# 	xx = np.array([x for x in xx if all([i > -7 for i in x])])
	# 	try:
	# 		model = LinearRegressionModel()
	# 		e = crossValidate(model, xx, Y, weighted=False)
	# 		print featureCandidates[f][0], model.theta, e
	# 	except Exception:
	# 		print 'Exception'
	# 		continue

	# Run forward search
	# features, testError, trainingError = forwardSearch(model, X, Y, cvMethod='simple', weighted=False)
	# print len(featureCandidates), features
	# print [featureCandidates[i][0] for i in features]
	# print testError


	# Error-m relation
	ms = [30, 300, 3000, 30000]
	result = []
	for m in ms:
		_, testError, trainingError = forwardSearch(model, X[:m], Y[:m], cvMethod='simple')
		result.append((m, testError, trainingError))
		print m, testError, trainingError
	print result


if __name__ == '__main__':
	main(sys.argv)
