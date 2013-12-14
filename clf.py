"""
Try several classifiers with PCA / CCA
to visualize feature separation.
Since Gaussian works for continuous features and Multinomial
for discrete, can we try averaging the two perhaps?
"""

from utils import reader, mappings, extractor
import sys
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
# from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

# ---------------------------------------------------------
# Adjustable parameters

# Debugging: Controls how many lines reader reads in
LIMIT = None

# Number of buckets to divide dataset into for kfolding
KFOLD_LEVEL = 10

# Divide labels into buckets to get more consistent data
LABEL_BUCKET_SIZE = 20

# How many buckets to plot for the error graph
EG_N = 300

# ---------------------------------------------------------
# Extraction

def extractAge(line):
    return int(line[7:9])

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
        extractAge(line),
        extractProviderSeen(line),
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

#       extractPrescription(line)
    ]


# ----------------------------------------------------------

def pcaModel(X):
    """ Returns column of pca'd models"""
    pca = PCA(n_components=1)
    pca.fit(X)
    return pca.explained_variance_ratio_

def roundBinaryFeature(val):
    return 1 if val >= 0 else 0

def findNearestBucket(val):
    global LABEL_BUCKET_SIZE
    return (val / LABEL_BUCKET_SIZE) * LABEL_BUCKET_SIZE

def extractTimeWithMd(line):
    return int(line[291:293])

def extractTimeWithMdBucket(line):
    time = extractTimeWithMd(line)
    return findNearestBucket(time)

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

def extractLabel(line):
    """
    Main label extraction fn.
    """
    time = extractTimeWithMdBucket(line)
    if time > 60: time = 60
    return time

# ---------------------------------------------------------
# Main
def printResults(score, results, testY): 
    print "Score: ", 
    print "Prediction     |     Actual     |     Difference     "
    print "-----------------------------------------------------"
    for i in range(0, len(results)):
        pred = results[i]
        actual = testY[i]
        diff = results[i] - testY[i] 
        print "       %02d      |       %02d       |         %02d        " % (pred, actual, diff)
    print "Done"
    for i in range(0, len(results)):
        pred = results[i]
        actual = testY[i]
        diff = results[i] - testY[i] 
        print "%d,%d,%d" % (pred, actual, diff)

def plotErrorGraph(model, X, y, testX, testY, Y_raw, testY_raw):
    global EG_N
    """
    Prints the training - testing error graph found
    in Andrew Ng's practical ML slides.
    trainingError and testError are ([xs], [ys]) to plot.
    """
    trainErrors = []
    testErrors = []

    ms = [int(n) for n in np.linspace(30, 2000, num=EG_N, endpoint=False)]
    # ms = [int(n) for n in np.linspace(30, 10000, num=500, endpoint=False)]
    # ms += [int(n) for n in np.linspace(10000, len(y), num=, endpoint=False)]
    for i in ms:        
        print X[:i].shape, y[:i].shape
        
        Xys = np.hstack([X, y.reshape(-1,1), Y_raw.reshape(-1,1)])
        np.random.permutation(Xys)
        X = Xys[:, :-2]
        y = Xys[:, -2]
        Y_raw = Xys[:, -1]

        model.fit(X[:i], y[:i])
        trainErrors.append(1 - model.score(X[:i], y[:i]))
        testErrors.append(1 - model.score(testX[:i], testY[:i]))
        # trainErrors.append(calculateError(model.predict(X[:i]), y[:i], Y_raw[:i]))
        # testErrors.append(calculateError(model.predict(testX[:i]), testY[:i], testY_raw[:i]))
    
    # fig = plt.figure()
    # ax = fig.add_subplot(2,1,1)
    plt.plot(ms, trainErrors, 'g-')
    plt.plot(ms, testErrors, 'r-')
    # ax.set_xscale('log')
    plt.show()

def quickPlot(model, X, y):
    preds = model.predict(X)
    plt.hist(model.predict(X), max(preds) / 10)
    # plt.subplot(212)
    # plt.hist(y, max(y) / 10)
    plt.show()

def calculateError(prediction, y, raw):
    global LABEL_BUCKET_SIZE
    """ May modify predictions """
    assert(len(prediction) == len(y))
    assert(len(prediction) == len(raw))
    N = len(prediction)
    diff = np.abs(np.subtract(prediction, y))
    idxs = np.where(diff != 0)[0]
    for i in np.arange(10, 100, 10):
        prediction[np.where(prediction == i)] = i + (LABEL_BUCKET_SIZE / 2)
    diff = np.sum(np.abs(np.subtract(prediction[idxs], raw[idxs])))
    return float(diff) / float(N)

def selectFeatures(Model, X, y):
    model = Model()
    fsel = SelectPercentile(score_func=f_classif, percentile=5)
    fsel.fit(X, y)
    arr = fsel.get_support()
    print "features: ", np.where(arr == True)
    plt.hist(model.predict(X))
    plt.hist(y)
    plt.show()

def main(argv):
    raise Exception("Dont' call this from main. Instead, open up a Python interpreter in the project directory and type `from clf import *` ")

    if len(argv) < 3:
        print "Usage: python naive_bayes.py <train_data> <test_data>"
        sys.exit(1)

    y, X = reader.read(argv[1], **{
        'extractFeaturesFn': extractFeatures2009, 
        'extractLabelsFn': extractLabel, 
        'limit': LIMIT
    })

    testY, testX = reader.read(argv[2], **{
        'extractFeaturesFn': extractFeatures2010, 
        'extractLabelsFn': extractLabel, 
        'limit': LIMIT
    })

# ----------------------------------------------------------
# Exec

if __name__ == "__main__":
    main(sys.argv)
else:

    # ------- Actual ---------

    # setup for scripting
    y, X = reader.read("data/2009", **{
        'extractFeaturesFn': extractFeatures2009, 
        'extractLabelsFn': extractLabel, 
        'limit': LIMIT
    })
    
    # idxs = np.where(X[:, 3] == 0)
    # X = X[idxs]
    # y = y[idxs]

    # Process and remove drugs and procedures and 
    # leave it up to PCA to reduce dimensionality
    # procedures = X[:, 3:11]
    # drugTypes = X[:, 12:19]
    # X = np.delete(X, range(3, 19), axis=1)

    # idxs = np.where(y==10)[0][:25]
    # X = np.delete(X, idxs, axis=0)
    # y = np.delete(y, idxs, axis=0)

    # ------- Test -------

    testY, testX = reader.read("data/2010", **{
        'extractFeaturesFn': extractFeatures2010, 
        'extractLabelsFn': extractLabel, 
        'limit': LIMIT
    })

    # idxs = np.where(testX[:, 3] == 0)
    # testX = testX[idxs]
    # testY = testY[idxs]

    # Process and remove test procedures
    # testProcedures = testX[:, 3:11]
    # testDrugTypes = testX[:, 12:19]
    # testX = np.delete(testX, range(3, 19), axis=1)

    Y_raw = reader.read('data/2009', **{
        'extractFeaturesFn': extractFeatures2009,
        'extractLabelsFn': extractTimeWithMd,
        'limit': LIMIT
    })[0]

    # read actual y values
    testY_raw = reader.read("data/2010", **{
        'extractFeaturesFn': extractFeatures2010,
        'extractLabelsFn': extractTimeWithMd,
        'limit': LIMIT
    })[0]

    Y_raw[np.where(Y_raw > 60)] = 60
    testY_raw[np.where(testY_raw > 60)] = 60
    # testY_raw = testY_raw[idxs]

    # kmX = (reader.read("data/2009", **{
    #     'extractFeaturesFn': extractFeatures,
    #     'extractLabelsFn': extractLabel,
    #     'limit': LIMIT
    # }))[1]

    # Add clustering classification
    # km = KMeans(n_clusters=50).fit(kmX)
    # centers = km.fit_predict(X)
    # X = np.hstack([X, centers.reshape(-1,1)])

    # testCenters = km.fit_predict(testX)
    # testX = np.hstack([testX, testCenters.reshape(-1,1)])

    # ------- Run Algorithms -------

    print "------ Try MultinomialNB -------"
    nb = MultinomialNB()
    nb.fit(X, y)
    predictions = nb.predict(testX)
    print "Accuracy: ", nb.score(testX, testY)
    print "Error: ", calculateError(predictions, testY, testY_raw)

    print "------ Try Random Forests -------"
    rfc = RandomForestClassifier()
    rfc.fit(X, y)
    predictions = rfc.predict(testX)
    print "Accuracy: ", rfc.score(testX, testY)
    print "Error: ", calculateError(predictions, testY, testY_raw)

    # Calculate errors
    # prediction = nb.predict(testX)
    # raw = testY_raw
    # N = len(raw)
    # diff = np.abs(np.subtract(prediction, y))
    # idxs = np.where(diff != 0)[0]
    # for i in np.arange(10, 100, 10):
    #     prediction[np.where(prediction == i)] = i + 5
    # diff = np.sum(np.abs(np.subtract(prediction[idxs], raw[idxs])))
    # print float(diff) / float(N)

    # plotErrorGraph(rfc, X , y, testX, testY)

    # print "------- Try One VS One SVMs -------"
    # ovoc = OneVsOneClassifier(LinearSVC(random_state=0))
    # ovoc.fit(X, y)
    # print "Accuracy: ", ovoc.score(X, y)
    # print "Error: ", calculateError(ovoc, X, y)
