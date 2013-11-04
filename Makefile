RED = \033[0;31m
GREEN = \033[0;32m
END = \033[0m

.PHONY: test data

time_with_md_visualization:
	@python ./tools/parser.py ./data/2010
	@echo "${GREEN}Data parsed; visualizing ${END}"
	@python ./tools/time_with_md_visualizer.py ./data/2010.csv

