import sys
from utils import reader
import numpy as np
from scipy.linalg import eigh


class Model():
	def __init__(self, k):
		# Initialization here
		self.k = k
		self.pca = False
		pass


	def pca(self, x):
		x = np.float(x.T)
		NoOfExamples = x.shape[1]
		NoOfFeatures = x.shape[0]

		S = np.zeros([NoOfFeatures, NoOfFeatures])
		for example in range(NoOfExamples):
			x[:, example] = x[:, example] - np.average(x[:, example]) # normalize data
			sigma = np.sqrt(np.average(np.power(x[:, example], 2))) # normalize data
			x[:, example] = x[:, example] / sigma # normalize data

			S += np.dot(x[:, example], x[:, example].T) # build S matrix

		S = S / NoOfFeatures # finish building S matrix

		evalues, evectors = eigh(x, eigvals=(NoOfFeatures-self.k,NoOfFeatures-1)) # find the k top eigenvectors

		xReduced = np.dot(evectors.T, x) # calculate the input features in the low dimensional space

		self.pca = True

		return xReduced # Return features in low dimensionsal space. Number of rows is k and number of columns is the number of examples


def main(argv):
	if len(argv) < 3:
		print "Usage: python pca.py <train_data> <test_data>"
		sys.exit(1)

	y, x = reader.read(argv[1], extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)
	# testY, testX = reader.read(argv[2], extractFeaturesFn=extractFeatures, extractLabelsFn=extractLabel, limit=LIMIT)

	k = 3

	model = Model(k)
	compressedFeatures = model.pca(x)


if __name__ == "__main__":
	main(sys.argv)	