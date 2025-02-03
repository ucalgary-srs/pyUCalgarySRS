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

import os
import datetime
import warnings
import pytest
from pyucalgarysrs import Calibration, SRSError, Data
from ...conftest import find_dataset

# globals
DATA_DIR = "%s/../../../test_data/read_calibration" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "REGO_flatfield_15649_20211019-+_v02.sav",
        "dataset_name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
        "expected_params": {
            "version": "v02",
            "start": datetime.datetime(2021, 10, 19),
            "stop": None,
            "detector_uid": "15649",
        },
    },
    {
        "filename": "REGO_flatfield_15653_20141002-20191212_v01.sav",
        "dataset_name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
        "expected_params": {
            "version": "v01",
            "start": datetime.datetime(2014, 10, 2),
            "stop": datetime.datetime(2019, 12, 12),
            "detector_uid": "15653",
        },
    },
    {
        "filename": "REGO_Rayleighs_15649_20141015-20211018_v01.sav",
        "dataset_name": "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
        "expected_params": {
            "version": "v01",
            "start": datetime.datetime(2014, 10, 15),
            "stop": datetime.datetime(2021, 10, 18),
            "detector_uid": "15649",
        },
    },
    {
        "filename": "REGO_Rayleighs_15649_20211019-+_v02.sav",
        "dataset_name": "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
        "expected_params": {
            "version": "v02",
            "start": datetime.datetime(2021, 10, 19),
            "stop": None,
            "detector_uid": "15649",
        },
    },
])
@pytest.mark.data_read
def test_read_calibration_single_file(srs, capsys, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # read file
    data = srs.data.read(dataset, "%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check return type
    assert isinstance(data, Data) is True
    assert isinstance(data.data, list) is True
    for item in data.data:
        assert isinstance(item, Calibration) is True

    # check return values
    assert data.data[0].detector_uid == test_dict["expected_params"]["detector_uid"]
    assert data.data[0].version == test_dict["expected_params"]["version"]
    assert data.data[0].generation_info.valid_interval_start == test_dict["expected_params"]["start"]
    assert data.data[0].generation_info.valid_interval_stop == test_dict["expected_params"]["stop"]
    assert data.data[0].dataset == dataset
    if ("FLATFIELD" in test_dict["dataset_name"]):
        assert data.data[0].flat_field_multiplier is not None
        assert data.data[0].rayleighs_perdn_persecond is None
    if ("RALEIGHS" in test_dict["dataset_name"]):
        assert data.data[0].rayleighs_perdn_persecond is not None
        assert data.data[0].flat_field_multiplier is None

    # check __str__ and __repr__ for Data type
    print_str = str(data)
    assert print_str != ""
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check __str__ and __repr__ for Calibration type
    print_str = str(data.data[0])
    assert print_str != ""

    # check __str__ and __repr__ for CalibrationGenerationInfo type
    print_str = str(data.data[0].generation_info)
    assert print_str != ""

    # check pretty print methods
    data.data[0].pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""
    data.data[0].generation_info.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "REGO_flatfield_15649_20211019-+_v02.sav",
            "REGO_flatfield_15653_20141002-20191212_v01.sav",
        ],
        "dataset_name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
    },
    {
        "filenames": [
            "REGO_Rayleighs_15649_20141015-20211018_v01.sav",
            "REGO_Rayleighs_15649_20211019-+_v02.sav",
        ],
        "dataset_name": "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
    },
])
@pytest.mark.data_read
def test_read_calibration_multiple_files(srs, all_datasets, test_dict, capsys):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read files
    data = srs.data.read(dataset, file_list)

    # check return type
    assert isinstance(data, Data) is True
    assert isinstance(data.data, list) is True
    for item in data.data:
        assert isinstance(item, Calibration) is True

    # check __str__ and __repr__ for Data type
    print_str = str(data)
    assert print_str != ""
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "REGO_flatfield_15649_20211019-+_v02.sav",
        ],
        "dataset_name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
        "n_parallel": 1,
    },
    {
        "filenames": [
            "REGO_flatfield_15649_20211019-+_v02.sav",
            "REGO_flatfield_15653_20141002-20191212_v01.sav",
        ],
        "dataset_name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
        "n_parallel": 2,
    },
    {
        "filenames": [
            "REGO_Rayleighs_15649_20141015-20211018_v01.sav",
        ],
        "dataset_name": "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
        "n_parallel": 2,
    },
    {
        "filenames": [
            "REGO_Rayleighs_15649_20141015-20211018_v01.sav",
            "REGO_Rayleighs_15649_20211019-+_v02.sav",
        ],
        "dataset_name": "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
        "n_parallel": 2,
    },
])
@pytest.mark.data_read
def test_read_calibration_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.read(
        dataset,
        file_list,
        n_parallel=test_dict["n_parallel"],
    )

    # check return type
    assert isinstance(data, Data) is True
    assert isinstance(data.data, list) is True
    for item in data.data:
        assert isinstance(item, Calibration) is True


@pytest.mark.data_read
def test_read_calibration_badperms_file(srs):
    # set filename
    f = "%s/REGO_Rayleighs_15651_20210908-+_v02.sav" % (DATA_DIR)
    os.chmod(f, 0o000)

    # read file and check problematic files (not quiet mode)
    with pytest.raises(SRSError) as e_info:
        srs.data.readers.read_calibration(f, quiet=False)
    assert "Error reading calibration file" in str(e_info)

    # read file and check problematic files (quiet mode)
    with pytest.raises(SRSError) as e_info:
        srs.data.readers.read_calibration(f, quiet=True)
    assert "Error reading calibration file" in str(e_info)

    # change perms back
    os.chmod(f, 0o644)


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "REGO_flatfield_15649_20211019-+_v02.sav",
        ],
        "dataset_name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
        "n_parallel": 1,
    },
])
@pytest.mark.data_read
def test_read_calibration_startend(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    with warnings.catch_warnings(record=True) as w:
        # read file
        data = srs.data.read(
            dataset,
            file_list,
            n_parallel=test_dict["n_parallel"],
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
        )

    # check return type
    assert isinstance(data, Data) is True
    assert isinstance(data.data, list) is True
    for item in data.data:
        assert isinstance(item, Calibration) is True

    # check that the warning appeared
    assert len(w) == 1
    assert issubclass(w[-1].category, UserWarning)
    assert "Reading of calibration files does not support the start_time or end_time parameters." in str(w[-1].message)
