#!/usr/bin/python
from utils.reader import read
from utils.mappings import features
from utils.extractor import extract
import numpy as np

def writeline(file, line):
	return file.write(line + "\n")

def extractTimeWithMd(line):
	return int(line[291:293])

def extractFeatures2010(line):
	"""
	Extract features based on specs from 2010
	"""
	return [extract(line, spec) for _, spec in features["2010"]]

def extractLabel(line):
	"""
	Main label extraction fn.
	"""
	return extractTimeWithMd(line)

y, X  = read("./data/2010", **{
	'extractFeaturesFn': extractFeatures2010, 
	'extractLabelsFn': extractLabel
})

for i in range(0, X.shape[1]):
	feature = X[:, i]
	featureName = features["2010"][i][0]
	print "Writing feature for " + featureName
	file = open("./results/featureAverages/%d-%s.txt" % (i, featureName), "w")
	writeline(file, "FeatureValue   |   Average TimeWithMD")
	distinctFeature = set(list(feature))
	for value in distinctFeature:
		average = np.average([y[i] for i in np.where(feature == value)])
		writeline(file, "       %s      |       %s      " % (value, average))
	print "Done"
	file.close()