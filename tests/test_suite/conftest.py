# Copyright 2024 University of Calgary
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import shutil
import copy
import pytest
import pyucalgarysrs
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--api-url", action="store", default="https://api-staging.phys.ucalgary.ca", help="A specific API URL to use")
    parser.addoption("--api-key", type=str, help="A specific API key to use")


@pytest.fixture(scope="session")
def api_url(request):
    return request.config.getoption("--api-url")


@pytest.fixture
def api_key(request):
    """
    NOTE: we aren't using the api key functionality quite yet, but including the plumbing
    anyways.
    """
    return request.config.getoption("--api-key")


@pytest.fixture(scope="function")
def srs(api_url):
    return pyucalgarysrs.PyUCalgarySRS(api_base_url=api_url)


@pytest.fixture(scope="session")
def all_datasets(api_url):
    srs = pyucalgarysrs.PyUCalgarySRS(api_base_url=api_url)
    return srs.data.list_datasets()


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    glob_str = "%s/pyucalgarysrs_data_*testing*" % (str(Path.home()))
    path_list = sorted(glob.glob(glob_str))
    for p in path_list:
        shutil.rmtree(p)


def find_dataset(datasets, dataset_name):
    for d in datasets:
        if (d.name == dataset_name):
            return copy.deepcopy(d)
    return None
