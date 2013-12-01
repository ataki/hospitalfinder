"""
Generic extractor with a list of fields for extraction.
mapping_tup should be a tuple of (start_index, end_index, typefn)
"""
import re

def booleanSmooth(value):
	return 0 if value != 1 else 1

def extract(line, name, mapping_tup):
	# process type
	start, end, typefn = mapping_tup
	if len(line) < end:
		raise IndexError("End out of bounds")
	if len(line) < start:
		raise IndexError("Start out of bounds")
	try:
		# if typefn is str:
		# 	raw = line[start:end].strip()
		# 	if len(raw) != 0:
		# 		value = int(re.sub("[^0-9]", "", raw))
		# 		if value < 0: value = 0
		# 	else:
		# 		value = 0
		if typefn is int or typefn is float:
			value = typefn(line[start:end])
			if value < 0: value = 0
			if typefn is float:
				value = int(value)
		elif typefn is bool:
			value = int(line[start:end])
			if value < 0: value = 0
		else:
			value = typefn(line[start:end])
	except ValueError:
		raise ValueError("Feature at (%d, %d) is %s and doesn't match %s" % (start, end, line[start:end], str(typefn)))
	
	# process value based on name
	if name in ["dayofweek", "injury", "isPrimaryPhysician", "isReferred", "seenBefore", "majorReason",
		"hasComputerForPatientInfo", "eveningWeekendVistsAllowed", "hasComputerForPrescriptionOrders", "hasComputerForViewingLabTestOrders",
		"hasComputerForViewingImageResults", "hasPlansForNewEMRIn18Months"]:
		value = booleanSmooth(value)
	elif name == "age":
		value = float(value) * float(value)
	elif name == "numberOfNewMedicationsCoded":
		value = 1 if value == 8 else 0
	elif name == "isReferredSeenBeforeCombo":
		isReferred = booleanSmooth(int(value[0:2]))
		seenBefore = booleanSmooth(int(value[2:3]))
		value = int(str(isReferred) + str(seenBefore))
	elif name in ["revenueFromPatientPayments", "revenueFromPrivateInsurance"]:
		value = 1 if value == 4 else 0
	
	if value < 0:
		print name
	return value
