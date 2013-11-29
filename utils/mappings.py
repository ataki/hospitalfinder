# 2010 Data Mapping
target = {
	"2010": ("timeWithMD", (291, 293, int)),
	"2009": ("timeWithMD", (291, 293, int))
}

features = {

	# ------

	"2010": [
		# ("dayOfWeek", (6, 7, int)),
		# ("age", (7, 10, int)),
		# ("sex", (10, 11, int)),
		# ("ethnicity", (11, 13, int)),
		# ("race", (13, 15, int)),
		# ("paytype", (19, 21, int)),
		# ("tobacco", (25, 27, int)),
		# ("injury", (27, 29, int)),
		# ("isPrimaryPhysician", (45, 47, int)),
		("isReferred", (47, 49, int)),
		("seenBefore", (49, 50, int)),
		# ("isReferredSeenBeforeCombo", (47, 50, str)),
		("pastVisitsLastYear", (50, 52, int)),
		# ("majorReason", (52, 54, int)),
		# ("height", (94, 96, int)),
		# ("weight", (96, 99, int)),
		# ("bmi", (99, 105, float)),
		# ("temperature", (105, 109, float)),
		# ("hadService", (115, 116, bool)),
		# ("totalServices", (203, 205, int)),
		# ("healthEducation", (205, 206, bool)),
		# ("totalHealthEducationOrdered", (216, 218, int)),
		# ("medicationProvided", (218, 219, int)),
		# ("numberOfNewMedicationsCoded", (275, 276, int)),
		# ("numberOfContinuedMedicationsCoded", (276, 277, int)),
		# ("providersSeenNumberOfMedicationsCoded", (277, 278, bool)),
		# ("region", (299, 300, int)),
		# ("isMetropolitan", (300, 301, bool)),
		# ("typeOfDoctor", (304, 305, int)),
		# ("physicianCode", (305, 309, int)),
		# ("drug1Type", (319, 325, str)),
		# ("drug2Type", (376, 382, str)),
		# ("drug3Type", (433, 439, str)),
		# ("drug4Type", (490, 496, str)),
		# ("drug5Type", (547, 553, str)),
		# ("drug6Type", (604, 610, str)),
		# ("drug7Type", (661, 667, str)),
		# ("drug8Type", (718, 724, str)),
		# ("officeSetting", (775, 776, int)),
		# ("soloPractice", (776, 778, bool)),
		# ("physicianEmploymentStatus", (778, 780, int)),
		# ("typeOfPracticeOwner", (780, 782, int)),
		# ("eveningWeekendVistsAllowed", (782, 784, int)),
		# ("hasComputerForPatientInfo", (798, 800, int)),
		# ("hasComputerForPrescriptionOrders", (808, 810, int)),
		# ("hasComputerForLabTestOrders", (814, 816, int)),
		# ("hasComputerForViewingLabTestOrders", (818, 820, int)),
		# ("hasComputerForViewingImageResults", (824, 826, int)),
		# ("hasPlansForNewEMRIn18Months", (857, 859, int)),
		# ("revenueFromMedicare", (859, 861, int)),
		# ("revenueFromMedicaid", (861, 863, int)),
		# ("revenueFromPrivateInsurance", (863, 865, int)),
		("revenueFromPatientPayments", (865, 867, int)),
		# ("revenueFromManagedCareContacts", (871, 873, int)),
		# ("revenueFromUsualService", (873, 875, int)),
		# ("acceptingNewPatients", (883, 885, int)),
		# ("routineAppointmentSetupTime", (904, 906, int)),
		# ("whoCompletedForm", (972, 974, int)),
	],

	# ------

	"2009": [
		# ("dayOfWeek", (6, 7, int)),
		# ("age", (7, 10, int)),
		# ("sex", (10, 11, int)),
		# ("ethnicity", (11, 13, int)),
		# ("race", (13, 15, int)),
		# ("paytype", (19, 21, int)),
		# ("tobacco", (25, 27, int)),
		# ("injury", (27, 29, int)),
		# ("isPrimaryPhysician", (45, 47, int)),
		("isReferred", (47, 49, int)),
		("seenBefore", (49, 50, int)),
		# ("isReferredSeenBeforeCombo", (47, 50, str)),
		("pastVisitsLastYear", (50, 52, int)),
		# ("majorReason", (52, 54, int)),
		# ("height", (94, 96, int)),
		# ("weight", (96, 99, int)),
		# ("bmi", (99, 105, float)),
		# ("temperature", (105, 109, float)),
		# ("hadService", (115, 116, bool)),
		# ("totalServices", (203, 205, int)),
		# ("healthEducation", (205, 206, bool)),
		# ("totalHealthEducationOrdered", (216, 218, int)),
		# ("medicationProvided", (218, 219, int)),
		# ("numberOfNewMedicationsCoded", (275, 276, int)),
		# ("numberOfContinuedMedicationsCoded", (276, 277, int)),
		# ("providersSeenNumberOfMedicationsCoded", (277, 278, bool)),
		# ("region", (299, 300, int)),
		# ("isMetropolitan", (300, 301, bool)),
		# ("typeOfDoctor", (304, 305, int)),
		# ("physicianCode", (305, 309, int)),
		# ("drug1Type", (319, 325, str)),
		# ("drug2Type", (376, 382, str)),
		# ("drug3Type", (433, 439, str)),
		# ("drug4Type", (490, 496, str)),
		# ("drug5Type", (547, 553, str)),
		# ("drug6Type", (604, 610, str)),
		# ("drug7Type", (661, 667, str)),
		# ("drug8Type", (718, 724, str)),
		# ("officeSetting", (775, 776, int)),
		# ("soloPractice", (776, 778, bool)),
		# ("physicianEmploymentStatus", (778, 780, int)),
		# ("typeOfPracticeOwner", (780, 782, int)),
		# ("eveningWeekendVistsAllowed", (782, 784, int)),
		# ("hasComputerForPatientInfo", (798, 800, int)),
		# ("hasComputerForPrescriptionOrders", (802, 804, int)),
		# ("hasComputerForLabTestOrders", (808, 810, int)),
		# ("hasComputerForViewingLabTestOrders", (814, 816, int)),
		# ("hasComputerForViewingImageResults", (816, 818, int)),
		# ("hasPlansForNewEMRIn18Months", (830, 832, int)),
		# ("revenueFromMedicare", (832, 834, int)),
		# ("revenueFromMedicaid", (834, 836, int)),
		# ("revenueFromPrivateInsurance", (836, 838, int)),
		("revenueFromPatientPayments", (838, 840, int)),
		# ("revenueFromManagedCareContacts", (844, 846, int)),
		# ("revenueFromUsualService", (846, 848, int)),
		# ("acceptingNewPatients", (883, 885, int)),
		# ("routineAppointmentSetupTime", (904, 906, int)),
		# ("whoCompletedForm", (972, 974, int)),
	]
}