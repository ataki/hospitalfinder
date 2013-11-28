from utils.reader import read
from utils.mappings import features
from utils.extractor import extract
import numpy as np

outfile = open('./2010.csv', 'w')

def writeline(arr):
	global outfile
	return outfile.write(",".join(arr) + "\n")

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

writeline([spec[0] for spec in features["2010"]] + ['timeWithMD(y)'])
for example in np.hstack([X, y.reshape(-1,1)]):
	writeline([str(a) for a in example])

outfile.close()