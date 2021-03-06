"""
Library for common reading functions for data.

Returns a tuple (y, x)
y is a numpy.array of training data.
x is a numpy.array of training examples.
Takes in 

`path_to_data_file` Absolute path

`extractFeaturesFn(line)` Given a string representing a line, returns 
 an array corresponding to a set of features

`extractLabel(line)` Given a string representing a line, returns
a value corresponding to a labeling

Usage:

    import reader    
    y, x = reader.read(path_to_data_file, extractFeaturesFn, extractLabelFn)

"""
import sys
from numpy import array

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
            if limit != None and counter >= limit:
                break
            else:
                x.append(extractFeaturesFn(line))
                y.append(extractLabelsFn(line))
                counter += 1

    return (array(y), array(x))

