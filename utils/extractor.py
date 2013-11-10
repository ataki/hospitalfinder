"""
Generic extractor with a list of fields for extraction.
"""

def extractor(line, start, end, typefn):
	"""
	Extrator doesn't take into account that some
	lines may be shorter than others. 
	i.e. each training / test example you pass
	in must have the same amount of features
	"""
	if len(line) < end:
		print "End out of bounds"
		raise IndexError
	if len(line) < start:
		print "Start out of bounds"
		raise IndexError
	return typefn(line[start:end])

def extractorHandleNull(line, start, end, typefn):
	"""
	Extractor, unlike above, doesn't assume that 
	lines are the same. Calls the above function, but on 
	an IndexError, simply sets the feature to None
	to indicate the non-existence of that feature
	"""
	try:
		return extractor(line, start, end, typefn)
	except IndexError:
		return None
