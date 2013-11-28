"""
Try several classifiers with PCA / CCA
to visualize feature separation.
Since Gaussian works for continuous features and Multinomial
for discrete, can we try averaging the two perhaps?
"""

from utils import reader, mappings, extractor
import sys
import numpy as np
from math import floor
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

# ---------------------------------------------------------
# Adjustable parameters

# Debugging: Controls how many lines reader reads in
LIMIT = None

# Number of buckets to divide dataset into for kfolding
KFOLD_LEVEL = 10

# Divide labels into buckets to get more consistent data
LABEL_BUCKET_SIZE = 25

# Optimal set of features
IDEAL_FEATURES = [11, 14, 16, 28, 29]

# How many steps to plot for the error graph
EG_N = 200

# ---------------------------------------------------------
# Extraction

def roundBinaryFeature(val):
	return 1 if val >= 0 else 0

def findNearestBucket(val):
	global LABEL_BUCKET_SIZE
	return (val / LABEL_BUCKET_SIZE) * LABEL_BUCKET_SIZE

def extractTimeWithMd(line):
	time = int(line[291:293])
	return findNearestBucket(time)

def extractFeatures2010(line):
	"""
	Extract features based on specs from 2010
	"""
	return [extractor.extract(line, name, spec) for name, spec in mappings.features["2010"]]

def extractFeatures2009(line):
	"""
	Extract features based on specs from 2009
	"""
	return [extractor.extract(line, name, spec) for name, spec in mappings.features["2009"]]

def extractLabel(line):
	"""
	Main label extraction fn.
	"""
	return extractTimeWithMd(line)

# ---------------------------------------------------------
# Main
def printResults(score, results, testY): 
	print "Score: ", 
	print "Prediction     |     Actual     |     Difference     "
	print "-----------------------------------------------------"
	for i in range(0, len(results)):
		pred = results[i]
		actual = testY[i]
		diff = results[i] - testY[i] 
		print "       %02d      |       %02d       |         %02d        " % (pred, actual, diff)
	print "Done"
	for i in range(0, len(results)):
		pred = results[i]
		actual = testY[i]
		diff = results[i] - testY[i] 
		print "%d,%d,%d" % (pred, actual, diff)

def plotErrorGraph(model, X, y, testX, testY):
	global EG_N
	"""
	Prints the training - testing error graph found
	in Andrew Ng's practical ML slides.
	trainingError and testError are ([xs], [ys]) to plot.
	"""
	trainErrors = []
	testErrors = []
	ms = [int(n) for n in np.linspace(30, len(y), num=EG_N, endpoint=False)]
	for i in ms:
		print X[:i].shape, y[:i].shape
		model.fit(X[:i], y[:i])
		trainErrors.append(1 - model.score(X, y))
		testErrors.append(1 - model.score(testX, testY))
	plt.plot(ms, trainErrors, 'g-')
	plt.plot(ms, testErrors, 'r-')
	plt.show()

def selectFeatures(Model, X, y):
	model = Model()
	fsel = SelectPercentile(score_func=f_classif, percentile=5)
	fsel.fit(X, y)
	arr = fsel.get_support()
	print "features: ", np.where(arr == True)
	plt.hist(model.predict(X))
	plt.hist(y)
	plt.show()

def main(argv):
	if len(argv) < 3:
		print "Usage: python naive_bayes.py <train_data> <test_data>"
		sys.exit(1)

	y, X = reader.read(argv[1], **{
		'extractFeaturesFn': extractFeatures2009, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})
	# X = X[:, IDEAL_FEATURES]

	testY, testX = reader.read(argv[2], **{
		'extractFeaturesFn': extractFeatures2010, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})
	# testX = testX[:, IDEAL_FEATURES]

	nb = NaiveBayes()
	nb.fit(X, y)
	plotErrorGraph(nb, X, y, testX, testY)

def script():
	y, X = reader.read("./data/2009", **{
		'extractFeaturesFn': extractFeatures2009, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})
	testY, testX = reader.read("./data/2010", **{
		'extractFeaturesFn': extractFeatures2010, 
		'extractLabelsFn': extractLabel, 
		'limit': LIMIT
	})

	X = np.vstack([X, testX])
	y = np.concatenate([y, testY], axis=0)

	nb = NaiveBayes()
	nb.train(X, y)

	print "NB Score: ", nb.score(X,y)

# ----------------------------------------------------------
# Exec

if __name__ == "__main__":
	main(sys.argv)
else:
	script(sys.argv)