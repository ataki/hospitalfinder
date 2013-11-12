"""
Generic extractor with a list of fields for extraction.
mapping_tup should be a tuple of (start_index, end_index, typefn)
"""

def extract(line, mapping_tup):
	start, end, typefn = mapping_tup
	
	if len(line) < end:
		raise IndexError("End out of bounds")
	if len(line) < start:
		raise IndexError("Start out of bounds")
	
	try:
		value = typefn(line[start:end])
	except ValueError:
		raise ValueError("Feature at (%d, %d) is %s and doesn't match %s" % (start, end, line[start:end], str(typefn)))
	return value
