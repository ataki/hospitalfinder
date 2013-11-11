"""
Generic extractor with a list of fields for extraction.
mapping_tup should be a tuple of (start_index, end_index, typefn)
"""

def extract(line, mapping_tup):
	"""
	Extrator doesn't take into account that some
	lines may be shorter than others. 
	i.e. each training / test example you pass
	in must have the same amount of features
	"""
	start, end, typefn = mapping_tup
	if len(line) < end:
		print "End out of bounds"
		raise IndexError
	if len(line) < start:
		print "Start out of bounds"
		raise IndexError
	return typefn(line[start:end])

def extractHandleNull(line, mapping_tup):
	"""
	Extractor, unlike above, doesn't assume that 
	lines are the same. Calls the above function, but on 
	an IndexError, simply sets the feature to None
	to indicate the non-existence of that feature
	"""
	try:
		return extract(line, mapping_tup)
	except IndexError:
		return None
