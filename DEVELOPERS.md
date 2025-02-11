# Development

Some common things you can do:
- `make update` Update the Python dependency libraries
- `tools/bump_version.py` Bump the version number
- `make test` Run the linting tests
- `make test-pytest` Run the functionality tests
- `make docs` Generate pdoc documentation

## Setup

Clone the repository and install primary and development dependencies using Poetry.

```console
$ git clone git@github.com:ucalgary-srs/pyUCalgarySRS.git
$ cd pyUCalgarySRS
$ conda activate <env name>
$ pip install poetry
$ poetry install
```

## Documentation

Documentation for the PyUCalgarySRS library is contained within this repository. To generate the docs, run the following:

```console
$ make docs
```

## Testing

PyUCalgarySRS includes several test evaluations bundled into two groups: linting and functionality tests. The linting includes looking through the codebase using tools such as Ruff, Pycodestyle, PyRight, and Bandit. The functionality tests use PyTest to test modules and functions in the library.

There exist several makefile targets to help run these tests quicker/easier. Below are the available commands:

- `make test` Run all linting tests
- `make test-linting` Run all linting tests
- `make test-pytest` Run all automated functional tests
- `make test-coverage` View test coverage report (must be done after `make test-pytest` or other coverage command)

The PyTest functionality tests include several categories of tests. You can run each category separately if you want using the "markers" feature of PyTest. All markers are found in the pytest.ini file at the root of the repository or by running `poetry run pytest --markers`.

Below are some more commands for advanced usages of PyTest.

- `poetry run pytest -v` Run all tests in verbose mode
- `poetry run pytest -s` Run all tests and capture stdout (ie. print messages)
- `poetry run pytest --collect-only` List all available tests
- `poetry run pytest --markers` List all markers (includes builtin, plugin and per-project ones)
- `cat pytest.ini` List custom markers

You can also run Pytest against a different API. By default, it runs against the staging API, but you can alternatively tell it to run against the production API, or a local instance.

- `poetry run pytest --api-url=https://api.phys.ucalgary.ca` Run all tests against production API
- `poetry run pytest --api-url=http://localhost:3000` Run all tests against a local instance of the API
- `poetry run pytest --help` View usage for pytest, including the usage for custom options (see the 'custom options' section of the output)

Below are some more commands for evaluating the PyTest coverage.

- `poetry run coverage report` View test coverage report
- `poetry run coverage html` Generate an HTML page of the coverage report
- `poetry run coverage report --show-missing` View the test coverage report and include the lines deemed to be not covered by tests

Note that the coverage report only gets updated when using the Makefile pytest targets, or when running coverage manually with the necessary options. More information about usage of the `coverage` command can be found [here](https://coverage.readthedocs.io).

## Publishing new release

To publish a new release, you must set the PyPI token first within Poetry and then upload the new package:

```console
$ poetry config pypi-token.pypi <pypi token>
$ make publish
```

## Development Notes

### Code coverage

The overall mentality is that the coverage report is in place to help us quickly find out any new holes in the test suite. The below two rules help us with this, indicating that any file showing <100% coverage has a new hole introduced that we were not previously aware of.

  1. Lines marked with `# pragma: nocover-ok` indicate that they are excluded from the coverage report and that we have consciously done so. These lines have been verified to ensure that it's ok that we're not paying attention to the test suite holes they are attributed to
  2. Lines marked with `# pragma: nocover` indicate that they are excluded from the coverage report, but that we should work towards plugging the holes.

The example notebooks CAN be included in the coverage by including `--cov=pyucalgarysrs --cov-report= --cov-append` in the pytest call from the Makefile. However, we currently prefer to not include them in the coverage since the regular test suite is needed anyways, and will be more comprehensive. Notebooks are tested before each release is published, to ensure they are working.
