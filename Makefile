RED = \033[0;31m
GREEN = \033[0;32m
END = \033[0m

.PHONY: test data

data:
	@python ./parsers/pre_run.py ./data/2010
	@echo "${GREEN}Data parsed; visualizing ${END}"

	@python ./tools/visualizer.py ./data.csv

