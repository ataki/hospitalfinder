"""
Sample code to get started reading in data in kmeans
"""
import sys
from utils import reader

# Limit on the number of lines read in by reader
LIMIT = 100

# ---------------------------------------------------------
# Model

class Model():
	def __init__(self):
		# Initialization here
		pass

	def train(self, x, y):
		self.x = x
		self.y = y

		# process data here
		
		pass

# ---------------------------------------------------------
# Extraction

def extractFeatures(line):
	return line[8:10]

def extractLabel(line):
	return line[9:11]

# ---------------------------------------------------------
# Main

def main(argv):
	if len(argv) < 3:
		print "Usage: python naive_bayes.py <train_data> <test_data>"
		sys.exit(1)

	y, x = reader.read(argv[1], extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)
	testY, testX = reader.read(argv[2], extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)

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
