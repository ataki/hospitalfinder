"""
Library for common reading functions for data.

Usage:

    import reader

    # Returns a tuple (y, x)
    # y is an array of training data.
    # x is an array of training examples.

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

def read(filename, extractFeaturesFn=None, extractLabelFn=None):

    # labels
    y = []

    # features
    x = []

    if not extractFeaturesFn:
        extractFeaturesFn = extractFeatures

    if not extractFeaturesFn:
        extractLabelFn = extractLabel

    with open(filename, 'r') as f:
        for line in f:
            x.append(extractFeaturesFn(line))
            y.append(extractLabelFn(line))

    return (y, x)

