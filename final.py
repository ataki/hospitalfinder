"""
Place for finalized model.
Usage:
    python final.py <training_data> <testing_data>
"""
from utils import extractor, mappings, reader
import sys
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn import feature_selection, cross_validation
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

# Debugging: Controls how many lines reader reads in
LIMIT = None

# Number of buckets to divide dataset into for kfolding
KFOLD_LEVEL = 10

# Divide labels into buckets to get more consistent data
LABEL_BUCKET_SIZE = 10

def findNearestBucket(val):
    # Assign correct bucket for categories
    global LABEL_BUCKET_SIZE
    return val - (val % LABEL_BUCKET_SIZE)

def roundToNearestBucket(ys):
    return np.array([findNearestBucket(y) for y in ys])

def extractTimeWithMd(line):
    # Extract field rounded down to nearest bucket size
    return int(line[291:293])

def extractFeatures(line):
    """
    Extract features based on specs from 2009
    """
    return [extractor.extract(line, spec) for _, spec in mappings.features["2009"]]

def extractLabel(line):
    """
    Main label extraction fn.
    """
    return extractTimeWithMd(line)

def main(argv):
    print "Please run for now as an import into ipython"
    sys.exit(0)

# import os
# def generateFileName():
#   existing = sorted(os.listdir("./results"), key=str.lower, reverse=True)
#   for file in existing:
#       parts = file.split(".")
#       if len(parts) >= 2 and len(parts[0]) >= 6 and parts[:5] == "final":
#           return "final%s.txt" % parts[6]
#   return "final1.txt"

def extractFeatures2010(line):
    """
    Extract features based on specs from 2010
    """
    return [extractor.extract(line, name, spec) for name, spec in mappings.features["2010"]]

def extractFeatures2009(line):
    """
    Extract features based on specs from 2009
    """
    return [extractor.extract(line, name, spec) for name, spec in mappings.features["2009"]]

def cv(model, X, Y):
    # Cross validation
    testErrors = []
    trainingErrors = []
    predictions = []

    kf = cross_validation.KFold(len(Y), n_folds=10)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)
        for i in range(len(Y_train)):
            prediction = model.predict(X_train[i])
            trainingErrors.append(np.absolute(prediction - Y_train[i]))
            predictions.append(prediction)
        # print 'training error: ', np.mean(np.absolute(model.predict(X_train) - Y_train))
        for i in range(len(Y_test)):
            prediction = model.predict(X_test[i])
            testErrors.append(np.absolute(prediction - Y_test[i]))
            predictions.append(prediction)
        # print 'test error: ', np.mean(np.absolute(model.predict(X_test) - Y_test))

    print 'avg training error: ', np.mean(trainingErrors)
    print 'avg test error: ', np.mean(testErrors)

    # plt.figure()
    # plt.hist(testErrors, 30)
    # plt.show()
    # plt.figure()
    # plt.hist(trainingErrors, 30)
    # plt.show()
    # plt.figure()
    # plt.hist(predictions)
    # plt.show()

def printValueDistribution(arr):
    distincts = set(arr)
    for val in distincts:
        print "%d : %d" % (val, len(np.where(arr == val)[0])) 

def fsel(model, X, Y):
    # Feature selection
    selector = feature_selection.RFECV(model)
    selector = selector.fit(X, Y)
    # print selector.support_
    # print selector.ranking_
    # print selector.grid_scores_
    # X = selector.transform(X)
    return selector

if __name__ == "__main__":
    # main(sys.argv)
    print WARNING + "Don't call this via the command line; instead, open up ipython and type in `from final import *`"
    sys.exit(1)
else:

    # -----------------------------------------------------------

    print OKBLUE
    print "Reading in data"
    print ENDC

    y, X = reader.read("data/2009", **{
        'extractFeaturesFn': extractFeatures2009, 
        'extractLabelsFn': extractLabel, 
        'limit': LIMIT
    })
    testY, testX = reader.read("data/2010", **{
        'extractFeaturesFn': extractFeatures2010, 
        'extractLabelsFn': extractLabel, 
        'limit': LIMIT
    })

    print OKGREEN
    print "Done reading data"
    print ENDC

    # -----------------------------------------------------------
    # Preprocess for linear regression

    XY = np.array([xy for xy in np.hstack((X, y.reshape(-1, 1)))
            if all([i > -7 for i in xy])])
    XY = np.random.permutation(XY)
    cleanX = XY[:, :-1]
    cleanY = XY[:, -1]

    print OKGREEN
    print "Done post-processing"
    print ENDC

    # -----------------------------------------------------------
    # Preprocess for classification
    clfFeatures = ['isReferred', 'seenBefore', 'pastVisitsLastYear',
        'revenueFromPatientPayments']

    trainM = mappings.features["2009"]
    featureMappings = [i for i in range(0, len(trainM))
        if trainM[i][0] in clfFeatures]

    clfX = X[:, featureMappings]
    clfY = roundToNearestBucket(y)
    clfTestX = testX[:, featureMappings]
    clfTestY = roundToNearestBucket(testY)

    # -----------------------------------------------------------
    # Build Models

    print OKBLUE
    print "Building models"
    print ENDC

    clf = MultinomialNB()
    clf.fit(clfX, clfY)

    linreg = RidgeCV(normalize=True, alphas=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 10, 100, 1000, 10000, 100000])
    # Feature selection, increases performance a lot
    selector = fsel(linreg, X, y)
    newX = selector.transform(X)
    newTestX = selector.transform(testX)
    linreg.fit(newX, y)

    # Error over m
    # for m in [10, 20, 50, 100] + range(200, len(y), 200):
    #     # print 'm = %s' % m
    #     cv(model, X[:m], Y[:m])

    # cv(model, X, Y)
    print OKGREEN
    print "Done building models"
    print ENDC

    # -----------------------------------------------------------
    # Predictions

    print OKBLUE
    print "Making predictions"
    print ENDC

    clfP = clf.predict(clfTestX)
    linP = linreg.predict(newTestX)

    print OKGREEN
    print "Done making predictions"
    print ENDC

    # -----------------------------------------------------------
    # Analyze residuals
    print OKGREEN
    print "Analyzing residuals: "
    print "The following variables shall be defined."
    print "Indices in various array correspond to one another"
    print "------------------------------------------------------------------------------"
    print "msp_LinearRegression_i   : indices of mispredictions > 5"
    print "msp_LinearRegression_y   : mispredicted labels"
    print "msp_LinearRegression     : mispredictions"
    print "msp_Classification_i     : indices of misclassified examples"
    print "msp_Classification_y     : mispredicted labels"
    print "msp_Classification       : mispredictions"
    print ""
    print "msp_Common_i             : indices of common mispredictions"
    print "msp_Common_y             : mispredicted labels common to both"
    print "------------------------------------------------------------------------------"
    print ENDC

    _lin_diff = np.abs(np.subtract(linP, testY))
    msp_LinearRegression_i = np.where(_lin_diff > 5)[0]
    msp_LinearRegression_y = testY[msp_LinearRegression_i]
    msp_LinearRegression = _lin_diff[msp_LinearRegression_i]

    _clf_diff = np.abs(np.subtract(clfP, testY))
    msp_Classification_i = np.where(_clf_diff > 0)[0]
    msp_Classification_y = testY[msp_Classification_i]
    msp_Classification = _clf_diff[msp_Classification_i]

    A = set(msp_Classification_i)
    B = set(msp_LinearRegression_i)
    msp_Common_i = list(A & B)
    msp_Common_y = testY[msp_Common_i]

    msp_Uncommon_i = list(A.difference(B))
    msp_Uncommon_y = testY[msp_Uncommon_i]

    

