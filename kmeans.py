"""
Sample code to get started reading in data in kmeans
"""
import sys
from utils import reader
import numpy as np

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

def extractSex(line):
	return int(line[8:10])

def extractPrescription(line):
	return int(line[400:401])

def extractFeatures(line):
	return [
		extractSex(line),
		extractPrescrption(line)
	]

def extractLabel(line):
	return int(line[9:11])

# ---------------------------------------------------------
# Main

def main(argv):
	if len(argv) < 3:
		print "Usage: python kmeans.py <train_data> <test_data>"
		sys.exit(1)

	y, x = reader.read(argv[1], extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)
	testY, testX = reader.read(argv[2], extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)

	model = Model()
	model.train(x, y)

	print x[3,0]
	

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
