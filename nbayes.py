"""
Milestone 1:

Implements Naive Bayes Classifier for hospital data.
Exports a Model class useful for training data sets.

We use discretized NB via binning to get optimal
performance (see Hsu, C., H. Huang, and T. Wong. 2000. 
Why Discretization Works for Naive Bayesian Classifiers).
"""

from utils import reader, mappings, extractor
import sys
import copy
import numpy as np
from math import log, exp, floor

# Debugging: Controls how many lines reader reads in
LIMIT = 50

# Number of buckets to divide dataset into for kfolding
KFOLD_LEVEL = 10

# Divide labels into buckets to get more consistent data
LABEL_BUCKET_SIZE = 10

# ---------------------------------------------------------
# Model

class Model(object):

	def __init__(self):
		self.y = None
		self.x = None
		self.labels = None
		self.pCount = None
		self.numExamples = None
		self.featureValues = None
		self.yUnique = None
		self.pFeatures = None
		self.pLabels = None

	def train(self, x, y):
		self.x = x
		self.y = y
		self.numExamples = len(self.y)

		# helps with cross-validation.
		# aggregateRows is X with Y col tacked on
		self.aggregateRows = np.hstack( (self.x, self.y.reshape(-1,1)) )

		# Array of arrays of distinct values.
		# Each entry corresponds to a feature
		self.featureValues = [np.unique(col) for col in self.x.T]

		# categories: array
		# distinct values that y takes on
		self.yUnique, self.pCount= self._countCategories()

		# p_labels: array
		# each entry gives the probability of the corresponding label
		self.pLabels = [float(count) / float(self.numExamples) for count in self.pCount]

		# p_features: map of arrays
		# each value is an array of featureMaps
		# each featureMap gives a mapping of {featureValue: probabilityOfOccurence}
		self.pFeatures = {}
		for cat in self.yUnique:
			self.pFeatures[cat] = self._countFeaturesForCategory(cat)

	def classify(self, example):
		"""
		Classifies according to max likelihood.
		Outputs a tuple of (prediction, max_likelihood)
		"""
		max_likelihood = 0
		prediction = None
		for label in self.yUnique:
			likelihood = self._calculateLikelihood(example, label)
			if likelihood >= max_likelihood:
				max_likelihood = likelihood
				prediction = label
		return (prediction, max_likelihood)

	def predict(self, testExamples, testCategories):
		"""
		Returns an int representing the error in predictions.
		Calculate error by averaging predictions i.e. treat the 
		prediction as a linear regression problem and calculate
		squares error
		"""
		global LABEL_BUCKET_SIZE
		
		sum = 0
		maxDiff = max(testCategories)

		for example, correctLabel in zip(testExamples, testCategories):
			decision, likelihood = self.classify(example)
			if decision != correctLabel:
				if type(correctLabel) == int or type(correctLabel) == float:
					error = abs(float(correctLabel) - float(decision)) / maxDiff
					sum += pow(error, 2)
				else:
					sum += 1
		return float(sum) / float(len(testCategories))

	def crossValidate(self, method):
		"""
		Returns the estimated generalization error based on cross
		validation.

		Method: the cross validation scheme to use (kfold|simple)
		Raises NotImplementedError if the method is not valid

		Usage
		model.crossValidate('simple')
		model.crossValidate('k-fold')
		"""
		global KFOLD_LEVEL

		trainX = None
		trainY = None
		testX = None
		testY = None

		if method == 'simple':
			trainX, trainY, testX, testY = self._getCrossValidationSimpleData()
			self.train(trainX, trainY)
			return self.predict(testX, testY)

		elif method == 'kfold':
			permutation = np.random.permutation(self.aggregateRows)
			return np.average([self._getCrossValidationForBucket(i, permutation) for i in range(0, KFOLD_LEVEL)])
		
		else:
			raise NotImplementedError

	def _countCategories(self):
		"""
		Returns a tuple of uniqueKeys, counts.
		Each uniqueKey is a unique category;
		the corresponding entry in counts is 
		the number of occurrences
		"""
		uniqueKeys = np.unique(self.y)
		bins = uniqueKeys.searchsorted(self.y)
		return uniqueKeys, np.bincount(bins)

	def _countFeatureKey(self, value, featureCol, numDistinctFeatureValues):
		"""
		Given a distinct feature value, counts the number of occurrences
		of that value in the given feature column.
		Uses Laplace Smoothing
		"""
		occurrences = float(len(featureCol[featureCol == value]) + 1)
		total = float(len(featureCol) + numDistinctFeatureValues)
		return occurrences / total 

	def _createFeatureMap(self, values, featureCol):
		"""
		Given a distinct set of feature values, returns a map where
		each k-v pair map is the probability that the 
		given feature takes on that value.
		"""
		featureMap = {}
		numDistinctFeatureValues = len(values)
		for v in values:
			featureMap[v] = self._countFeatureKey(v, featureCol, numDistinctFeatureValues)
		return featureMap

	def _countFeaturesForCategory(self, label):
		"""
		Returns [ { featureMap1 }, { featureMap2 }, ... ]
		where each entry maps a featureValue to 
		its probability of occurring
		"""
		indices = np.where(self.y == label)
		examples = self.x[indices]
		# Iterate over the ith feature column
		# For each feature, returns a map of the feature to the 
		# probability of that feature occurring.
		featureMaps = []
		for i in range(0, len(self.featureValues)):
			values = self.featureValues[i]
			featureCol = examples[:, i]
			featureMaps.append(self._createFeatureMap(values, featureCol))
		return featureMaps

	def _calculateLikelihood(self, example, label):
		"""
		Given a training vector (example) and a classified 
		"""
		# Takes exp(sum(logs(probabilities)))
		log_likelihood = float(0)
		featureMappings = self.pFeatures[label]
		for i in range(0, len(example)):
			mapping = featureMappings[i]
			value = example[i]
			try:
				prob = mapping[value]
			except KeyError:
				prob = float(1) / float(len(mapping.keys()))
			assert(prob != 0)
			log_likelihood += log(prob)
		return exp(log_likelihood)

	def _getCrossValidationSimpleData(self):
		"""
		Simple CV splits dataset into 70% / 30%.
		It then trains on 70% and tests with 30%
		"""
		permutation = np.random.permutation(self.aggregateRows)
		stop = int(len(permutation) * 0.7)
		trainRows = permutation[0:stop]
		testRows = permutation[stop:]
		trainX = np.delete(trainRows, -1, axis=1)
		trainY = trainRows[:, 1]
		testX = np.delete(testRows, -1, 1)
		testY = testRows[:, 1]
		return (trainX, trainY, testX, testY)

	def _getCrossValidationForBucket(self, ith, permutation):
		trainX, trainY, testX, testY = self._getCrossValidationKFoldData(ith, permutation)
		self.train(trainX, trainY)
		return self.predict(testX, testY)

	def _getCrossValidationKFoldData(self, ith, permutation):
		global KFOLD_LEVEL

		bstart = ith * permutation
		bend = (ith + 1) * permutation

		trainRows = np.hstack( (permutation[0:bstart], permutation[bend:len(permutation)]) ) 
		testRows = permutation[bstart, bend]

		trainX = np.delete(trainRows, -1, 1)
		trainY = trainRows[:, 1]
		testX = np.delete(testRows, -1, 1)
		testY = testRows[:, 1]
		return (trainX, trainY, testX, testY)

# ---------------------------------------------------------
# Extraction

def findNearestBucket(val):
	# Assign correct bucket for categories
	global LABEL_BUCKET_SIZE
	return val - (val % LABEL_BUCKET_SIZE)

def extractTimeWithMd(line):
	# Extract field rounded down to nearest bucket size
	time = int(line[291:293])
	return findNearestBucket(time)

def extractFeatures2010(line):
	"""
	Extract features based on specs from 2010
	"""
	return [extractor.extract(line, spec) for _, spec in mappings.features["2010"]]

def extractFeatures2009(line):
	"""
	Extract features based on specs from 2009
	"""
	return [extractor.extract(line, spec) for _, spec in mappings.features["2009"]]

def extractLabel(line):
	"""
	Main label extraction fn.
	"""
	return extractTimeWithMd(line)

# ---------------------------------------------------------
# Feature Selection

def featureSelect(method, data, threshold):
	"""
	Returns tuple of (bestFeatures, bestError).
	
	bestFeature is an array of indices corresponding
	to columns in the data; it tells you which set of
	columns make for the best featuref
	bestError is the CV error corresponding to running
	simple cross validation on bestFeatures
	
	`method` technique to use (forward|backward)
	`data` tuple (y, x) of input data
	`threshold` desired length of featureSet
	"""
	y, x = data
	features = range(0, x.shape[1])  # set of features for inner for loop
	model = Model()

	if method == "forward":
		featureSet = []
		while len(featureSet) != threshold:
			bestFeature = None
			error = 1
			for i in features:
				model.train(np.take(x, featureSet + [i], axis=1), y)
				cvError = model.crossValidate('simple')
				if error > cvError:
					error = cvError
					bestFeature = i
			featureSet.append(bestFeature)
			features.remove(bestFeature)

	elif method == "backward":
		featureSet = copy.deepcopy(features)
		while len(featureSet) > threshold:
			worstFeature = 0 # by default, remove first feature if all else is equal
			error = 0
			for i in featureSet:
				model.train(np.take(x, [j for j in featureSet if i != j], axis=1), y)
				cvError = model.crossValidate('simple')
				if error < cvError:
					error = cvError
					worstFeature = i
			featureSet.remove(worstFeature)
	else:
		raise NotImplementedError

	return (featureSet, error)

# ---------------------------------------------------------
# Main

def main(argv):
	if len(argv) < 3:
		print "Usage: python naive_bayes.py <train_data> <test_data>"
		sys.exit(1)

	# Train on 2009 format
	y, x = reader.read(argv[1], **{
		'extractFeaturesFn': extractFeatures2009, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})

	# Test on 2010 format
	testY, testX = reader.read(argv[2], **{
		'extractFeaturesFn': extractFeatures2010, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})

	# Test Basic Predict
	# model = Model()
	# model.train(x, y)
	# error = model.predict(testX, testY)
	# print "Error: %f" % error

	# Feature Selection

	print "forward search: ", featureSelect('forward', (y, x), 10)
	print "backward search: ", featureSelect('backward', (y, x), 10)

if __name__ == "__main__":
	# ---
	# Executes as main script

	main(sys.argv)
else:
	# ---
	# Execute inside ipython

	DEFAULT_TRAIN = './data/2009'
	DEFAULT_TEST = './data/2010'
	
	y, x = reader.read(DEFAULT_TRAIN, **{
		'extractFeaturesFn': extractFeatures2009, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})

	testY, testX = reader.read(DEFAULT_TEST, **{
		'extractFeaturesFn': extractFeatures2010, 
		'extractLabelsFn': extractLabel, 
		'limit': floor(LIMIT / 4)
	})

