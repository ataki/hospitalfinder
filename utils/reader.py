"""
Library for common reading functions for data.

Usage:

    import reader

    # Returns a tuple (y, x)
    # y is an array of training data.
    # x is an array of training examples.
    
    (y, x) = reader.read(path_to_data_file)

"""

import datetime

# --------------------------------------------------------
# Helpers

def transformDttm(month, year):
    """
    (month, date) -> datetime("YYYY-MM-15 00:00:00")
    """
    return datetime.datetime(int(year), int(month), 15, 0, 0, 0, 0)

def extractDatetime(line):
    month = line[0:2]
    year = line[2:6]
    return transformDttm(month, year)

def extractDayOfWeek(line):
    return line[6]

def extractAge(line):
    return line[7:10]

def extractSex(line):
    sex = line[10]
    return "F" if int(sex) == 1 else "M"

def extractEthnicity(line):
    code = int(line[11:13])
    if code == -9:
        return "NULL"
    elif code == 1:
        return "Hisp/L"
    elif code == 2:
        return "Nonhisp/L"

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

def read(filename):

    # labels
    y = []

    # features
    x = []

    with open(filename, 'r') as f:
        for line in f:
            x.append(extractFeatures(line))
            y.append(extractLabel(line))

    return (y, x)

