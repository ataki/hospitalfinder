#!/usr/bin/python

"""
Milestone 1:

Implements Naive Bayes Classifier for hospital data.
"""

from utils import reader
# import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import sys

LIMIT = 1000

# ---------------------------------------------------------
# Model

class Model(object):

	def __init__(self):
		self.y = None
		self.x = None

		self.yUnique = None

		self.pFeatures = None
		self.pLabels = None

	def countLabels(self):
	    uniqueKeys = np.unique(self.y)
	    bins = uniqueKeys.searchsorted(self.y)
	    return uniqueKeys, np.bincount(bins)

	def countFeatureKey(self, key, featureCol):
		# Given a distinct value, counts the number of occurrences
		# of that value in the given feature column.
		return float(len(featureCol[featureCol == key])) / float(len(featureCol))

	def createFeatureMap(self, keys, featureCol):
		# Given a distinct set of keys, returns a map where
		# each k-v pair map is the probability that the 
		# given feature takes on that value
		featureMap = {}
		for k in keys:
			featureMap[k] = self.countFeatureKey(k, featureCol)
		return featureMap

	def countFeaturesForLabel(self, label):
		"""
		Returns [ featureMap ]
		where each entry maps a featureValue to 
		its probability of occurring
		"""
		indices = np.where(self.y == label)
		examples = self.x[indices]
		# Iterate over the ith feature column
		# For each feature, returns a map of the feature to the 
		# probability of that feature occurring.
		featureMaps = []
		for i in range(0, len(self.allFeatureKeys)):
			keys = self.allFeatureKeys[i]
			featureCol = examples[:, i]
			featureMaps.append(self.createFeatureMap(keys, featureCol))
		return featureMaps

	def train(self, x, y):
		self.x = x
		self.y = y
		self.numExamples = len(self.y)

		# Distinct feature values
		self.allFeatureKeys = [np.unique(col) for col in self.x.T]

		# labels: array.
		# distinct values that y takes on

		self.labels, self.pCount= self.countLabels()

		# p_labels: array. 
		# each entry gives the probability of the corresponding label
		self.pLabels = [float(count) / float(self.numExamples) for count in self.pCount]

		# p_features: array
		# an array of featureMaps for each entry. 
		# each featureMap gives a mapping from featureValue: probabilityOfOccurence
		self.pFeatures = [self.countFeaturesForLabel(label) for label in self.labels]

		print "Labels: ", self.labels
		print "pLabels: ", self.pLabels
		# print "pFeatures: ", self.pFeatures


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

def main(argv):
	if len(argv) < 2:
		print "Usage: python naive_bayes.py <csv_name>"

	y, x = reader.read(argv[1], extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)

	model = Model()
	model.train(x, y)

if __name__ == "__main__":
	main(sys.argv)
else:
	DEFAULT = './data/2010'
	y, x = reader.read(DEFAULT, extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)

	model = Model()
	model.train(x, y)

