"""
Weighted and unweighted linear regression
"""

from utils import reader, mappings, extractor
import sys
import numpy as np

# Debugging: Controls how many lines reader reads in
LIMIT = 100

class LinearRegressionModel(object):

	def __init__(self, X, Y):
		# Add intercept term to X
		self.X = np.column_stack((np.ones((X.shape[0], 1)), X))
		self.Y = Y
		self.theta = None

	def train(self, X, Y):
		pass

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
				diff = self.X[i, :] - x
				W[i][i] = np.exp(- np.dot(diff, diff) / (2 * np.square(tau)))

			# Calculate weighted theta using normal equation
			theta = np.dot(np.dot(np.dot(
				np.linalg.inv(np.dot(np.dot(self.X.T, W), self.X)),
				self.X.T), W), self.Y)

			return np.dot(theta, x)
		else:
			if self.theta == None:
				# Calcualte unweighted theta using normal equation
				self.theta = np.dot(np.dot(
					np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.Y)
			return np.dot(self.theta, x)


def extractFeatures2010(line):
	"""
	Extract features based on specs from 2010
	"""
	selectedFeatures = ["age", "sex"] # TODO: use feature selection algorithm
	return [extractor.extract(line, spec)
		for name, spec in mappings.features['2010']
		if name in selectedFeatures]

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

	model = LinearRegressionModel(X, Y)

	# Predict using age and sex
	print model.h(np.array([7, 1]))

if __name__ == '__main__':
	main(sys.argv)
