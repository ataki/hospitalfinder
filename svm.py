"""
Starter Code for SVM's. 
Will leverage scikit-learn's implementation
of SVM"s
"""

import sys
# from sklearn import svm
from utils import reader

# ---

LIMIT = 50

class Model(object):

	def __init__(self):
		self.x = None
		self.y = None

	def train(self, x, y):
		raise NotImplementedError

	def predict(self, testX, testY):
		raise NotImplementedError

# ---

def extractTimeWithMd(line):
	return float(line[291:293])

def extractFeatures(line):
	return []

def extractLabel(line):
	return extractTimeWithMd(line)

# ---

def main(argv):
    if len(argv) < 3:
        print "Usage: python naive_bayes.py <train_data> <test_data>"
        sys.exit(1)

    y, x = reader.read(argv[1], **{
        'extractFeaturesFn': extractFeatures, 
        'extractLabelsFn': extractLabel, 
        'limit': LIMIT
    })

    testY, testX = reader.read(argv[2], **{
        'extractFeaturesFn': extractFeatures, 
        'extractLabelsFn': extractLabel, 
        'limit': LIMIT
    })

    model = Model()
    model.train(x, y)
    error = model.predict(testX, testY)
    print "Error: %f" % error

# ---

if __name__ == "__main__":
    # Executes as main script
    # ...
    main(sys.argv)

