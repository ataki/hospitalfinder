RED = \033[0;31m
GREEN = \033[0;32m
END = \033[0m
HR = \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

YEAR = 2009
TEST_YEAR = 2010

.PHONY: test data

field_histogram:
	@echo "${HR}"
	@echo "Visualizes the field specified in tools/parser"
	@echo "${HR}"
	@python tools/parser data/${YEAR}
	@echo "${GREEN}Data parsed; constructing histogram ${END}"
	@python tools/field_visualizer data/${YEAR}.csv

nbayes:
	@echo "${HR}"
	@echo "Naive Bayes Train on ${YEAR} Data"
	@echo "${HR}"
	@python nbayes.py data/${YEAR} data/${TEST_YEAR}

clean: 
	@rm *.pyc
	@echo "Done"
