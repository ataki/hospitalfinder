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
		self.x = np.float_(x.T)
		NoOfExamples = self.x.shape[1]
		NoOfFeatures = self.x.shape[0]

		S = np.zeros([NoOfFeatures, NoOfFeatures])
		for example in range(NoOfExamples):
			S += np.dot(x[:, example], x[:, example].T)

		S = S / NoOfFeatures

		evalues, evectors = eigh(self.x, eigvals=(NoOfFeatures-self.k,NoOfFeatures-1))

		self.pca = True

		return evectors


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