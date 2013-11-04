"""
Milestone 1:

Implements Naive Bayes Classifier for hospital data.
"""

from ... import reader
import matplotlib.pyplot as plt
import numpy
import sys

def main(argv):
	if len(argv) < 2:
		print "Usage: python naive_bayes.py <csv_name>"

	y, x = reader.read(argv[1])
	
if __name__ == "__main__":
	main(sys.argv)
