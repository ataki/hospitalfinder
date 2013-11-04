"""
Outputs data formatted according to the 2010 data into csv 
Usage: python .py <optional path_to_file (relative to current dir)>
"""

import sys
import os

# The CSV file this will output data to

outfile = open(os.path.join(os.getcwd(), 'data', '2010.csv'), 'w')

# csv's delimiter; change to whatever you want

DELIM = ","

# --- Helpers ---

MONTHS = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',\
                'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}

def transformMonth(mon):
    if mon in MONTHS:
        return MONTHS[mon] 
    else:
        return mon

def transformDttm(month, year):
    """
    (month, date) -> "YYYY-MM-15 00:00:00"
    We use default day 15 for middle of month,
    and 00:00:00 for beginning of day
    """
    return str(year) + "-" + str(month) + "-" + "15 00:00:00"

# --- Extractors ---
# Given a line, transforms these into numbers
# Consult the docYYYY.pdf record format for more info

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

# ...
# Too many fields to keep writing extracters. We should
# be doing MAX 20 features for now...

# --- Main ---

line_counter = 0

def printStats():
    global line_counter
    print "Lines processed: " + str(line_counter)

def _writeline(file, line):
    """
    Writes contents of line and inserts unix-conforming newline
    """
    file.write(line + "\n")

def writeOut(line):
    """
    global outfile - assumed to be an opened file to write out to
    """
    global outfile
    _writeline(outfile, line)

def processlineW(line):
    """
    Extract features from each line and write them to outfile
    """
    global line_counter

    fields = [
        str(extractTimeWithMd(line)),
        # ... too many fields to do them all here
    ]
    writeOut(DELIM.join(fields))
    line_counter += 1

# CSV column headers
# headers = [
#     "datetime",
#     "day_of_week",
#     "age",
#     "sex",
#     "ethnicity",
#     ""
# ]

# def writeHeaders():
#     global outfile
#     global headers

#     line = DELIM.join(headers)
#     writeOut(line)

def main(argv):
    global outfile

    cwd = os.getcwd()
    path = cwd

    if len(argv) >= 2:
        path = os.path.join(cwd, argv[1])
    
    with open(path) as f:
        for line in f:
            processlineW(line)

    # To be sure we're on the right track
    printStats()

if __name__ == "__main__":
    main(sys.argv)
