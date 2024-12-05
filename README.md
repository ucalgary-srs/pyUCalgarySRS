# PyUCalgarySRS

[![GitHub tests](https://github.com/ucalgary-srs/pyUCalgarySRS/actions/workflows/tests_default.yml/badge.svg)](https://github.com/ucalgary-srs/pyUCalgarySRS/actions/workflows/tests_default.yml)
[![PyPI version](https://img.shields.io/pypi/v/pyucalgarysrs.svg)](https://pypi.python.org/pypi/pyucalgarysrs/)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/pyucalgarysrs)](https://pypi.python.org/pypi/pyucalgarysrs/)

This is the Python library for retrieving and loading data provided by UCalgary Space Remote Sensing. More information including tutorials and dataset descriptions can be found on our open data platform at https://data.phys.ucalgary.ca.

> [!NOTE]
> This library is primarily a sub-dependency for higher-level data access and analysis support libraries [PyAuroraX](https://github.com/aurorax-space/pyaurorax) and [PyUCRio](https://github.com/ucalgary-srs/pyUCRio). We recommend using those libraries for accessing and analyzing optical (all-sky imager, spectrograph) data and riometer data, respectively.

PyUCalagarySRS currently supports Python 3.9+.

Some links to help:
- [PyUCalgarySRS API Reference](https://docs-pyucalgarysrs.phys.ucalgary.ca)
- [Crib sheets that utilize PyUCalgarySRS](https://data.phys.ucalgary.ca/working_with_data/index.html#crib-sheets)
- [UCalgary Space Remote Sensing Open Data Platform](https://data.phys.ucalgary.ca)
- [Usage examples](https://github.com/ucalgary-srs/pyUCalgarySRS/tree/main/tests/standalone_scripts)

## Installation

Installation can be done using pip:

```
pip install pyucalgarysrs
```

If you want the most bleeding edge version of PyUCalgarySRS, you can install it directly from the Github repository:

```console
$ git clone https://github.com/ucalgary-srs/pyUCalgarySRS.git
$ cd pyUCalgarySRS
$ pip install .
```

## Usage

Below is how the library can be imported:

```python
import pyucalgarysrs
srs = pyucalgarysrs.PyUCalgarySRS()
```

A large collection of examples can be found in the [`tests/standalone_scripts`](https://github.com/ucalgary-srs/pyUCalgarySRS/tree/main/tests/standalone_scripts) directory. 
