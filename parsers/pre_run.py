# Outputs things into csv
# Usage: python pre_run.py <optional path_to_file (relative to current dir)>

import sys
import os
from utils.data import transformDttm, transformMonth

# The CSV file this will output data to

outfile = open(os.path.join(os.getcwd(), 'data.csv'), 'w')

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

def extract_datetime(line):
    month = line[0:2]
    year = line[2:6]
    return transformDttm(month, year)

def extract_day_of_week(line):
    return line[6]

def extract_age(line):
    return line[7:10]

def extract_sex(line):
    sex = line[10]
    return "F" if int(sex) == 1 else "M"

def extract_ethnicity(line):
    code = int(line[11:13])
    if code == -9:
        return "NULL"
    elif code == 1:
        return "Hisp/L"
    elif code == 2:
        return "Nonhisp/L"

# ...
# Too many fields to keep writing extracters. We should
# be doing MAX 20 features for now...

# --- Main ---

int line_counter = 0

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
    gobal outfile
    _writeline(outfile, line)

def processlineW(line):
    """
    Extract features from each line and write them to outfile
    """
    global line_counter

    fields = [
        extract_datetime(line),
        extract_day_of_week(line),
        extract_age(line),
        extract_sex(line),
        extract_ethnicity(line),
        # ... too many fields to do them all here
    ]
    writeOut(DELIM.join(fields))
    line_counter += 1

# CSV column headers
headers = [
    "datetime",
    "day_of_week",
    "age",
    "sex",
    "ethnicity",
    # ... Too many fields here. Decide max 20
]

def writeHeaders():
    global outfile
    global headers

    line = DELIM.join(headers)
    writeline(line)

def main(argv):
    global outfile

    cwd = os.getcwd()
    path = os.path.join(cwd, 'data')
    if len(argv) > 2:
        path = os.path.join(cwd, argv[1])

    # writeHeaders()  # No need now
    
    with open(path) as f:
        for line in f:
            processlineW(line)

    # To be sure we're on the right track
    printStats()

if __name__ == "__main__":
    main(sys.argv)
