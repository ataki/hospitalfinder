HospitalFinder: CS 229 Final Project
---

Jim Zheng, Yannis Petousis, and Scott Cheng

Stanford University, CS 229 Fall 2013-14 Final Project. 

Summary
---

We are looking to use various ML techniques to predict time with md (in minutes) at hospitals.

Our data comes from the National Ambulatory Medical Care Survey, collected by the National Center for Health Statistics. This annual survey samples physician-paient interactions across
the US. Sampling was done using a "multi-stage probability design". For more information, including dataset features, please consult pdfs/docs2010.pdf or pdfs/docs09.pdf.

We train on historical data (1993-2009) and test using 2010 data. 

We use the scipy stack [http://www.scipy.org/stackspec.html]

We store our main implementations within scripts at the top level directory.

The following is our folder structure:

- Raw data stored in `/data`.
- Executable scripts in `/tools`
- Importable libraries in `/utils`
- Data documentation in `/pdfs`

Installation
---

Requirements: Python 2.7+ , pip, brew (on Mac OSX)

Have the requirements? Then

	pip install -r requirements.txt

Requirements.txt has all packages for ML libraries, visualizations, web framework, and profiler.

Usage
---

Run these in the root project directory

`make hist` plots a histogram for a single field. 

`make <algo_name>` outputs our tuned algorithms' results training with 2009 and testing with 2010 dataset. This is very useful for testing.
(e.g. `make nbayes`)

Methods Used 
---

- [ x ] Naive Bayes
- [ - ] Linear Regression
- [ - ] SVMs
- [ - ] KMeans
- [ - ] Feature Selection

Contributing
---

New algorithms define a model class, use reader.read to obtain a matrix of features and a vector of labels. 

The following presents starter code for a new model:

```python

from utils import reader
# ... other imports

class Model(object):

	def __init__(self):
		# ...

	def train(self, x, y):
		# ...

# ---

def extractFeatures(line):
	return [
		# ... call extractors here
	]

def extractLabel(line):
	# ... returns a label value ...

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
else:
	# Executes on import
	# ...

```
