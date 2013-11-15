"""
Feature selector

Usage:

	from fsel import forwardSearch
	features, error = forwardSearch(mlModel, X, Y, **options)
	# options is passed to crossValidate and mlModel.predict.

"""

import numpy as np

from cv import crossValidate

def forwardSearch(model, X, Y, threshold=0.01, **options):
	"""
	Run forward search on data samples X, Y using given model.
	Algorithm terminates when test error does not decrease after adding
	new features.

	Returns tuple of (bestFeatures, bestError).
	bestFeature is an array of indices of features.
	bestError is the cross validation error using bestFeatures

	threshold describes the termination condition, when the test error does
	not fall by threshold after adding a feature to model.
	options is passed to crossValidate and model.predict
	"""
	bestFeatures = []
	lastError = None
	features = range(X.shape[1]) # Feature indices

	while True:
		bestFeature = None
		bestError = float('inf')

		if not features:
			# Selected all features
			break

		for f in features:
			currentError = crossValidate(
				model, np.take(X, bestFeatures + [f], axis=1), Y, **options)
			if currentError < bestError:
				bestError = currentError
				bestFeature = f

		if lastError != None and (lastError - bestError) / lastError < threshold:
			break

		lastError = bestError
		bestFeatures.append(bestFeature)
		features.remove(bestFeature)

	return (bestFeatures, lastError)

def backwardSearch(model, X, Y, threshold=0.01, **options):
	# TODO
	pass
