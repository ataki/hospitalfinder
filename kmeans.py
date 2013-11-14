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
		self.k = 3
		self.MAXITER = 50
		self.trained = False
		pass

	def train(self, x, y):
		self.x = x
		self.y = y
		#self.centroids = 

		# process data here
		
		pass

	def test(self, tx, ty):
		self.tx = tx
		self.ty = ty
		#self.features = 

		# process data here

		pass



	def runKmeans(self, x):
		self.x = np.float_(x.T)
		self.EXAMPLE_LENGTH = self.x.shape[0]
		self.centroids = self.x[:, range(0, self.k)]
		centroidList = np.zeros([(self.x).shape[1], 1])

		convergence = False
		iteration = 0

		while convergence is not True:
			iteration += 1
			oldCentroids = np.array(self.centroids)
			for column in range(self.x.shape[1]):
				centroid = np.argmin(((self.centroids - self.x[:, column].reshape(self.EXAMPLE_LENGTH,1))**2).sum(axis=0))
				centroidList[column, 0] = centroid

			for centroidNo in range(self.k):
				logicVector = (centroidList==centroidNo)*1.
				count = logicVector.sum()
				if count == 0:
					self.centroids[:, centroidNo] = self.x[:, np.random.randint(0, self.x.shape[1])]
				else:
					self.centroids[:, centroidNo] = np.dot(self.x, logicVector).reshape(self.EXAMPLE_LENGTH) / count

			error = (np.sqrt(((self.x - self.centroids[:,map(int, centroidList)])**2).sum(axis=0))).sum()
			print error

			if iteration >= self.MAXITER or (np.abs(oldCentroids - self.centroids)).sum()==0: convergence = True

		self.trained = True

		print convergence
		print iteration


	def distanceToCentroids(self, x):
		assert self.trained

		x = np.float_(x.T)
		distances = np.zeros(x.shape[1] * self.k)
		ii = 0
		for example in range(x.shape[1]):
			euclidean = np.sqrt(((x[:, example].reshape(x.shape[0], 1) - self.centroids)**2).sum(axis=0))
			distances[(ii*self.k):((ii+1)*self.k)] = euclidean.sum() / self.k - euclidean
			distances = np.maximum(distances,0)
			ii = ii + 1

		print distances
		return distances


# ---------------------------------------------------------
# Extraction

#def extractAge(line):
#	return int(line[7:9])

def extractProviderSeen(line):
	return int(line[278:285])

def extractRegion(line):
	return int(line[299:300])

def extractProvider(line):
	return int(line[300:301])

def extractTypeDoctor(line):
	return int(line[304:305])

def extractTypeOffice(line):
	return int(line[775:776])

def extractSolo(line):
	return int(line[776:778])

def extractEmploymentStaus(line):
	return int(line[778:780])

def extractOwner(line):
	return int(line[780:782])

def extractWeekends(line):
	return int(line[782:784])

def extractNursingVisits(line):
	return int(line[784:786])

def extractHomeVisits(line):
	return int(line[786:788])

def extractHospVisits(line):
	return int(line[788:790])

def extractTelephoneConsults(line):
	return int(line[790:792])

def extractEmailConsults(line):
	return int(line[792:794])

def extractElectrBilling(line):
	return int(line[794:796])

def extractElectrMedRecords(line):
	return int(line[796:798])

def extractElectrPatProblems(line):
	return int(line[800:802])

def extractElectrPrescriptions(line):
	return int(line[802:804])

def extractElectrContraindications(line):
	return int(line[804:806])

def extractElectrPharmacy(line):
	return int(line[806:808])

def extractPercMedicare(line):
	return int(line[832:834])

def extractPercMediaid(line):
	return int(line[834:836])

def extractPercPrivIns(line):
	return int(line[836:838])

def extractPercPatientPay(line):
	return int(line[838:840])

def extractPercOther(line):
	return int(line[840:842])

def extractNumberManagedContracts(line):
	return int(line[842:844])

def extractFeatures(line):
	return [
		#extractAge(line),
		#extractProviderSeen(line),
		extractRegion(line),
		extractProvider(line),
		extractTypeDoctor(line),
		extractTypeOffice(line),
		extractSolo(line),
		extractEmploymentStaus(line),
		extractOwner(line),
		extractWeekends(line),
		extractNursingVisits(line),
		extractHomeVisits(line),
		extractHospVisits(line),
		extractTelephoneConsults(line),
		extractEmailConsults(line),
		extractElectrBilling(line),
		extractElectrMedRecords(line),
		extractElectrPatProblems(line),
		extractElectrPrescriptions(line),
		extractElectrContraindications(line),
		extractElectrPharmacy(line),
		extractPercMedicare(line),
		extractPercMediaid(line),
		extractPercPrivIns(line),
		extractPercPatientPay(line),
		extractPercOther(line),
		extractNumberManagedContracts(line),

#		extractPrescription(line)
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
	model.runKmeans(x)
	model.distanceToCentroids(x)

	print x
	print x.shape
	

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
