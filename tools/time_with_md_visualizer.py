import matplotlib.pyplot as plt
import numpy
import sys

def plot_count(fp):
	data = [int(l) for l in fp]

	counts = {}
	for point in data:
		if point in counts:
			counts[point] += 1
		else:
			counts[point] = 1
	
	print "Number of distinct points: ", len(counts.keys())	
	print "Mean", numpy.mean(data)
	print "Median", numpy.median(data)
	print "Standard deviation: ", numpy.std(data)

	print "    Time With MD     :    Count  	   "
	print "----------------------------------------"
	for k, v in counts.items():
		print "   %d    :     %d    " % (k, v) 

	plt.hist(data, 50)
	plt.show()

def main(argv):
	if len(argv) < 2:
		print "Usage: python visualizer.py <csv_name>"
	fp = open(argv[1], 'r')
	plot_count(fp)
	fp.close()

if __name__ == "__main__":
	main(sys.argv)