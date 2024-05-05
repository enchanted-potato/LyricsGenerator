lint:
	echo "Running formaters and linters"
	black .
	isort --skip .local --skip .poetry --skip .venv --profile black .