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

import pytest
import pyucalgarysrs

ALL_FILTER_TESTS = [
    {
        "name": "THEMIS",
        "expected_zero_results": False,
    },
    {
        "name": "REGO",
        "expected_zero_results": False,
    },
    {
        "name": "DAILY_KEOGRAM",
        "expected_zero_results": False,
    },
    {
        "name": "HOURLY_KEOGRAM",
        "expected_zero_results": False,
    },
    {
        "name": "KEOGRAM",
        "expected_zero_results": False,
    },
    {
        "name": "CALIBRATION",
        "expected_zero_results": False,
    },
    {
        "name": "SKYMAP",
        "expected_zero_results": False,
    },
    {
        "name": "",
        "expected_zero_results": False,
    },
    {
        "name": "some_bad_name",
        "expected_zero_results": True,
    },
]


@pytest.mark.data_datasets
def test_get_datasets(srs, capsys):
    # get all datasets
    datasets = srs.data.list_datasets()

    # check serialization
    for d in datasets:
        assert isinstance(d, pyucalgarysrs.Dataset) is True

    # check str method
    print_str = str(datasets[0])
    assert print_str != ""

    # check pretty print method
    datasets[0].pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.data_datasets
@pytest.mark.parametrize("test_dict", ALL_FILTER_TESTS)
def test_get_datasets_filter_name(srs, test_dict):
    # get datasets
    datasets = srs.data.list_datasets(name=test_dict["name"])

    # check if we expect any datasets to be found
    if (test_dict["expected_zero_results"] is True):
        assert len(datasets) == 0
        return
    else:
        assert len(datasets) > 0

    # check filter
    for d in datasets:
        assert test_dict["name"] in d.name


@pytest.mark.data_datasets
@pytest.mark.parametrize("test_dict", [
    {
        "name": "THEMIS_ASI_RAW",
        "error_expected": False
    },
    {
        "name": "TREX_RGB_RAW_NOMINAL",
        "error_expected": False
    },
    {
        "name": "SOME_BAD_DATASET",
        "error_expected": True
    },
])
def test_get_dataset(srs, test_dict):
    # get dataset
    if (test_dict["error_expected"] is False):
        dataset = srs.data.get_dataset(test_dict["name"])
        assert dataset.name == test_dict["name"]
    else:
        # we expect an error
        with pytest.raises(pyucalgarysrs.SRSAPIError) as e_info:
            dataset = srs.data.get_dataset(test_dict["name"])
        assert "Dataset not found" in str(e_info)
