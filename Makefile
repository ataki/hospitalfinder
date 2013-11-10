RED = \033[0;31m
GREEN = \033[0;32m
END = \033[0m
HR = ------------------------------------------

YEAR = 2009
TEST_YEAR = 2010

.PHONY: test data

hist:
	@echo "${HR}"
	@echo "Visualizes the field specified in tools/parser"
	@echo "${HR}"
	@python tools/parser data/${YEAR}
	@echo "${GREEN}Data parsed; constructing histogram ${END}"
	@python tools/histogram data/${YEAR}.csv

nbayes:
	@echo "${HR}"
	@echo "Naive Bayes Train on ${YEAR} Data, Test on ${TEST_YEAR} Data"
	@echo "${HR}"
	@python nbayes.py data/${YEAR} data/${TEST_YEAR}

kmeans:
	@echo "${HR}"
	@echo "Kmeans Train on ${YEAR} Data, Test on ${TEST_YEAR} Data"
	@echo "${HR}"
	@python kmeans.py data/${YEAR} data/${TEST_YEAR}

clean: 
	@rm *.pyc
	@echo "Done"
