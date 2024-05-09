.PHONY: install update get-test-data test test-linting test-ruff test-pycodestyle test-bandit test-pyright test-pytest test-pytest-production test-pytest-staging test-pytest-ci test-coverage clean publish

all:

install:
	pip install poetry
	poetry install
	${MAKE} get-test-data

update upgrade:
	pip install --upgrade poetry
	poetry update

clean:
	@rm -rf pyucalgarysrs.egg-info build dist

get-test-data:
	cd tests/test_data && rm -rf read_*
	cd tests/test_data && wget -O test_data.tar https://aurora.phys.ucalgary.ca/public/github_tests/pyucalgarysrs_test_data.tar
	cd tests/test_data && tar -xvf test_data.tar && rm test_data.tar

test: test-linting

test-linting: test-ruff test-pycodestyle test-pyright test-bandit

test-ruff ruff:
	@printf "Running ruff tests\n+++++++++++++++++++++++++++\n"
	ruff check --respect-gitignore --quiet pyucalgarysrs
	ruff check --respect-gitignore --quiet tests
	ruff check --respect-gitignore --quiet tools
	@printf "\n\n"

test-pycodestyle:
	@printf "Running pycodestyle tests\n+++++++++++++++++++++++++++\n"
	pycodestyle --config=.pycodestyle pyucalgarysrs
	pycodestyle --config=.pycodestyle tests
	pycodestyle --config=.pycodestyle tools
	@printf "\n\n"

test-pyright pyright:
	@printf "Running pyright tests\n+++++++++++++++++++++++++++\n"
	pyright
	@printf "\n\n"

test-bandit bandit:
	@printf "Running bandit tests\n+++++++++++++++++++++++++++\n"
	bandit -c pyproject.toml -r -ii pyucalgarysrs
	@printf "\n\n"

test-pytest pytest: test-pytest-staging

test-pytest-staging pytest-staging:
	pytest -n auto --cov=pyucalgarysrs --cov-report= --maxfail=1

test-pytest-production pytest-production:
	pytest -n auto --cov=pyucalgarysrs --cov-report= --maxfail=1 --api-url=https://api.phys.ucalgary.ca

test-pytest-ci pytest-ci:
	pytest -n auto -m "not data_datasets and not data_download and not data_geturls and not data_read" --cov=pyucalgarysrs --cov-report= --maxfail=1 --api-url=https://api.phys.ucalgary.ca

test-coverage coverage:
	coverage report

publish:
	${MAKE} test
	poetry build
	poetry publish
	${MAKE} clean
