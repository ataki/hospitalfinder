#!/usr/bin/python

"""
Milestone 1:

Implements Naive Bayes Classifier for hospital data.
"""

from utils import reader
# import matplotlib.pyplot as plt
import numpy as np
# import time
# import datetime
import sys
from math import log, exp

# Debugging: Controls how many lines reader reads in
LIMIT = 50

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

		# Array of arrays of distinct values.
		# Each entry corresponds to a feature
		self.featureValues = [np.unique(col) for col in self.x.T]

		# categories: array
		# distinct values that y takes on
		self.categories, self.pCount= self._countCategories()

		# p_labels: array
		# each entry gives the probability of the corresponding label
		self.pLabels = [float(count) / float(self.numExamples) for count in self.pCount]

		# p_features: map of arrays
		# each value is an array of featureMaps
		# each featureMap gives a mapping of {featureValue: probabilityOfOccurence}
		self.pFeatures = {}
		for cat in self.categories:
			self.pFeatures[cat] = self._countFeaturesForCategory(cat)

	def classify(self, example):
		"""
		Classifies according to max likelihood.
		Outputs a tuple of (prediction, max_likelihood)
		"""
		max_likelihood = 0
		prediction = None
		for label in self.featureValues:
			likelihood = self._calculateLikelihood(example, label)
			if likelihood > max_likelihood:
				max_likelihood = likelihood
				prediction = label
		return (prediction, max_likelihood)

	def test(self, testExamples, testCategories):
		error = float(0)
		for example, correctLabel in zip(testExamples, testCategories):
			decision, likelihood = self.classify(example)
			if decision != correctLabel:
				error += float(1)
		print "Training error: ", error / float(len(correctLabel))

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
		return float(len(featureCol[featureCol == value]) + 1) / float(len(featureCol) + numDistinctFeatureValues)

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
		# Takes exp(sum(logs(probabilities)))
		log_likelihood = float(0)
		featureMappings = self.pFeatures[label]
		for i in range(0, len(example)):
			mapping = featureMappings[i]
			value = example[i]
			prob = mapping[value]
			# can we ignore 0 values here?
			assert(prob != 0)
			log_likelihood += log(prob)
		return exp(log_likelihood)

# ---------------------------------------------------------
# Extraction

def extractDayOfWeek(line):
    return int(line[6])

def extractAge(line):
    return int(line[7:10])

def extractSex(line):
    # return "F" if int(sex) == 1 else "M"
    return 1 if int(line[10]) == 1 else 0

def extractTimeWithMd(line):
    return int(line[291:293])

def extractFeatures(line):
	return [
		extractDayOfWeek(line),
		extractAge(line),
		extractSex(line),
	]

def extractLabel(line):
	return extractTimeWithMd(line)

# ---------------------------------------------------------
# Main

def main(argv):
	if len(argv) < 2:
		print "Usage: python naive_bayes.py <train_csv> <test_csv>"

	y, x = reader.read(argv[1], extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)

	model = Model()
	model.train(x, y)

if __name__ == "__main__":
	main(sys.argv)
else:
	DEFAULT_TRAIN = './data/2009'
	DEFAULT_TEST = './data/2010'
	
	y, x = reader.read(DEFAULT_TRAIN, extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)

	model = Model()
	model.train(x, y)

	ty, tx = reader.read(DEFAULT_TEST, extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel)
	model.test()


