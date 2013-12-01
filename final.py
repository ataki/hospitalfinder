"""
Place for finalized model.

TODO: 
- [-] Incorporate kmeans centroids as features
- [-] Write code to clean both training and test data of outliers
- [-] Create sklearn pipeline of features
- [-] TODO: How to combine local weighted 

Usage:

	python final.py <training_data> <testing_data>

Will write an error summary report to results/finalN.txt
where N is the latest number of runs.
"""
from utils import extractor, mappings
import sys
import os
import numpy as np
import reader
import kmeans

# Debugging: Controls how many lines reader reads in
LIMIT = None

# Number of buckets to divide dataset into for kfolding
KFOLD_LEVEL = 10

# Divide labels into buckets to get more consistent data
LABEL_BUCKET_SIZE = 10

def roundBinaryFeature(val):
	return 1 if val >= 0 else 0

def findNearestBucket(val):
	# Assign correct bucket for categories
	global LABEL_BUCKET_SIZE
	return val - (val % LABEL_BUCKET_SIZE)

def extractTimeWithMd(line):
	# Extract field rounded down to nearest bucket size
	time = int(line[291:293])
	return findNearestBucket(time)

def extractFeatures(line):
	"""
	Extract features based on specs from 2009
	"""
	return [extractor.extract(line, spec) for _, spec in mappings.features["2009"]]

def extractLabel(line):
	"""
	Main label extraction fn.
	"""
	return extractTimeWithMd(line)

def main(argv):
	print "Processing features"
	if len(argv) < 3:
		print "Usage: python final.py <train_data> <test_data>"
		sys.exit(1)

	trainY, trainX = reader.read(argv[1], **{
		'extractFeaturesFn': extractFeatures, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})
	# Unsupervised learning pre-processing
	# linreg
	model = kmeans.Model()
	model.train(trainX, trainY)
	model.runKmeans(trainX)
	trainX = np.hstack(trainX, model.distanceToCentroids(trainX).reshape(-1, 1))
	# svm
	testY, testX = reader.read(argv[2], **{
		'extractFeaturesFn': extractFeatures, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})

def generateFileName():
	existing = sorted(os.listdir("./results"), key=str.lower, reverse=True)
	for file in existing:
		parts = file.split(".")
		if len(parts) >= 2 and len(parts[0]) >= 6 and parts[:5] == "final":
			return "final%s.txt" % parts[6]
	return "final1.txt"

if __name__ == "__main__":
	main(sys.argv)