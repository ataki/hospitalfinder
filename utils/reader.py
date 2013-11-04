"""
Library for common reading functions for data.

Usage:

    import reader

    # Returns a tuple (y, x)
    # y is a numpy.array of training data.
    # x is a numpy.array of training examples.

    # Takes in 
    # 
    # `path_to_data_file` Absolute path
    #
    # `extractFeaturesFn(line)` Given a string representing a line, returns 
    #  an array corresponding to a set of features
    #
    # `extractLabel(line)` Given a string representing a line, returns
    # a value corresponding to a labeling
    
    (y, x) = reader.read(path_to_data_file, extractFeaturesFn, extractLabelFn)

"""
import sys
from numpy import array

# --------------------------------------------------------
# Sample. Used as dummy defaults. 

def extractAge(line):
    return line[7:10]

def extractSex(line):
    sex = line[10]
    return "F" if int(sex) == 1 else "M"

def extractTimeWithMd(line):
    return int(line[291:293])

def extractFeatures(line):
    return [
        extractAge(line),
        extractSex(line),
    ]

def extractLabel(line):
    return extractTimeWithMd(line)

# -----------------------------------------------
# Main Exports

def printAndExit(msg):
    print msg
    sys.exit(1)

def read(filename, extractFeaturesFn=None, extractLabelsFn=None, limit=None):

    # labels
    y = []

    # features
    x = []

    if extractFeaturesFn == None:
        printAndExit("Reader requires a extractFeaturesFn")

    if extractLabelsFn == None:
        printAndExit("Reader requires a extractLabelsFn")

    counter = 0
    with open(filename, 'r') as f:
        for line in f:
            if limit != None and counter < limit:
                x.append(extractFeaturesFn(line))
                y.append(extractLabelsFn(line))
                counter += 1

    return (array(y), array(x))

