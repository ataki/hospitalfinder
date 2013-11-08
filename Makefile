RED = \033[0;31m
GREEN = \033[0;32m
END = \033[0m
HR = \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

.PHONY: test data

field_histogram:
	@echo "${HR}"
	@echo "Visualizes the field specified in tools/parser"
	@echo "${HR}"
	@python tools/parser data/2010
	@echo "${GREEN}Data parsed; constructing histogram ${END}"
	@python tools/field_visualizer data/2010.csv

nbayes:
	@echo "${HR}"
	@echo "Naive Bayes Train on 2010 Data"
	@echo "${HR}"
	@python naive_bayes.py data/2010

clean:
	@echo "li"
