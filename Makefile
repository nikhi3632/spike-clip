.PHONY: clean

# Remove Python bytecode caches and compiled files across the repo
clean:
	@echo "Cleaning __pycache__ directories and *.pyc/*.pyo files..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete
	@echo "Done."


