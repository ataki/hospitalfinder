"""
Feature selector

Usage:

	from fsel import forwardSearch
	features, error = forwardSearch(mlModel, X, Y, **options)
	# options is passed to crossValidate and mlModel.predict.

"""

import copy
import numpy as np

from cv import crossValidate

def forwardSearch(model, X, Y, threshold=0.0001, **options):
	"""
	Run forward search on data samples X, Y using given model.
	Algorithm terminates when test error does not decrease after adding
	new features.

	Returns tuple of (features, testError).
	features is an array of indices of selected features.
	testError is the cross validation error using selected features.

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
			# print "Looking at feature %d" % f
			try:
				filteredXY = np.array(
					[xy for xy in np.hstack((np.take(X, bestFeatures + [f], axis=1), Y.reshape(-1, 1)))
					if all([i > -7 for i in xy])])
				currentError = crossValidate(
					model, filteredXY[:, :-1], filteredXY[:, -1], **options)
			except Exception:
				# If exception happens, skip this feature
				# print 'Catched Exception'
				continue
			# print "Feature %d yields error %f, best error so far is %f" % \
			# 	(f, currentError, bestError), model.theta
			if currentError < bestError:
				bestError = currentError
				bestFeature = f
				trainingError = model.trainingError()
				# print 'Better feature: ', f

		if lastError != None and \
			(lastError == 0 or (lastError - bestError) / lastError < threshold):
			break

		lastError = bestError
		lastTrainingError = trainingError
		bestFeatures.append(bestFeature)
		features.remove(bestFeature)
		# print "Added feature %d" % bestFeature

	return (bestFeatures, lastError, lastTrainingError)

def backwardSearch(model, X, Y, threshold=0.0001, **options):
	# TODO
	pass
