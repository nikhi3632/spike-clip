SHELL := /bin/bash

.PHONY: clean data setup

# Setup virtual environment and install dependencies, then enter activated shell
setup:
	@python3 -m venv venv
	@bash -c "source ./venv/bin/activate && pip install -r requirements.txt && echo 'Setup complete. Entering activated shell...' && exec bash -i"

# Fetch data using fetch_data.py
data:
	@echo "Fetching data..."
	@python3 fetch_data.py

# Remove Python bytecode caches and compiled files across the repo
clean:
	@echo "Cleaning __pycache__ directories and *.pyc/*.pyo files..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete
	@echo "Done."


