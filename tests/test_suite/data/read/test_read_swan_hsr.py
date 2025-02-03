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
import numpy as np
from pathlib import Path
from pyucalgarysrs.data import ProblematicFile, HSRData
from ...conftest import find_dataset

# globals
DATA_DIR = "%s/../../../test_data/read_swan_hsr" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20240203_mean-hsr_k0_v01.h5",
        "expected_success": True,
        "expected_frames": 86220
    },
    {
        "filename": "20240204_mean-hsr_k0_v01.h5",
        "expected_success": True,
        "expected_frames": 86280
    },
])
@pytest.mark.data_read
def test_read_single_file(srs, all_datasets, test_dict, capsys):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # read file
    data = srs.data.read(dataset, "%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check data types
    assert isinstance(data.data, list) is True
    assert len(data.data) > 0
    for item in data.data:
        assert isinstance(item, HSRData) is True

    # check data
    assert data.data[0].raw_power.shape[1] == (test_dict["expected_frames"])
    assert data.data[0].raw_power.dtype == np.float32
    assert data.data[0].timestamp.shape == (test_dict["expected_frames"], )
    assert len(data.data[0].band_central_frequency) > 0
    assert len(data.data[0].band_passband) > 0
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}

    # check __str__ and __repr__ for Data type
    print_str = str(data)
    assert print_str != ""
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check __str__ and __repr__ for HSRData type
    print_str = str(data.data[0])
    assert print_str != ""
    data.data[0].pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
        ],
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_multiple_files(srs, capsys, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read files
    data = srs.data.read(dataset, file_list)

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert len(data.data) == len(test_dict["filenames"])
    for item in data.data:
        assert item.raw_power.dtype == np.float32
        assert item.raw_power.shape[1] == item.timestamp.shape[0]
        assert len(item.raw_power) != 0
        assert len(item.timestamp) != 0
        assert len(item.band_central_frequency) > 0
        assert len(item.band_passband) > 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}

    # check __str__ and __repr__ for Data type
    print_str = str(data)
    assert print_str != ""
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check __str__ and __repr__ for HSRData type
    print_str = str(data.data[0])
    assert print_str != ""
    data.data[0].pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20240203_mean-hsr_k0_v01.h5",
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filename": "20240204_mean-hsr_k0_v01.h5",
        "n_parallel": 2,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_single_file_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # read file
    data = srs.data.read(
        dataset,
        "%s/%s" % (DATA_DIR, test_dict["filename"]),
        n_parallel=test_dict["n_parallel"],
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert len(data.data) == 1
    for item in data.data:
        assert item.raw_power.dtype == np.float32
        assert item.raw_power.shape[1] == item.timestamp.shape[0]
        assert len(item.raw_power) != 0
        assert len(item.timestamp) != 0
        assert len(item.band_central_frequency) > 0
        assert len(item.band_passband) > 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 3,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_multiple_files_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.read(dataset, file_list, n_parallel=test_dict["n_parallel"])

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert len(data.data) == len(test_dict["filenames"])
    for item in data.data:
        assert item.raw_power.dtype == np.float32
        assert item.raw_power.shape[1] == item.timestamp.shape[0]
        assert len(item.raw_power) != 0
        assert len(item.timestamp) != 0
        assert len(item.band_central_frequency) > 0
        assert len(item.band_passband) > 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_pathlib_input(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append(Path(DATA_DIR) / Path(f))

    # read file
    data = srs.data.read(dataset, file_list, n_parallel=test_dict["n_parallel"])

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert len(data.data) == len(test_dict["filenames"])
    for item in data.data:
        assert item.raw_power.dtype == np.float32
        assert item.raw_power.shape[1] == item.timestamp.shape[0]
        assert len(item.raw_power) != 0
        assert len(item.timestamp) != 0
        assert len(item.band_central_frequency) > 0
        assert len(item.band_passband) > 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_first_record(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

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
            first_record=True,
        )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert len(data.data) == len(test_dict["filenames"])
    for item in data.data:
        assert item.raw_power.dtype == np.float32
        assert item.raw_power.shape[1] == item.timestamp.shape[0]
        assert len(item.raw_power) != 0
        assert len(item.timestamp) != 0
        assert len(item.band_central_frequency) > 0
        assert len(item.band_passband) > 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}

    # check that the warning appeared
    assert len(w) == 1
    assert issubclass(w[-1].category, UserWarning)
    assert "The 'first_record' parameter is not supported when reading SWAN HSR data." in str(w[-1].message)


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 3,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_no_metadata(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.read(
        dataset,
        file_list,
        n_parallel=test_dict["n_parallel"],
        no_metadata=True,
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert len(data.data) == len(test_dict["filenames"])
    for item in data.data:
        assert item.raw_power.dtype == np.float32
        assert item.raw_power.shape[1] == item.timestamp.shape[0]
        assert len(item.raw_power) != 0
        assert len(item.timestamp) != 0
        assert len(item.band_central_frequency) > 0
        assert len(item.band_passband) > 0

    # check the metadata
    assert len(data.metadata) == 0


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "some_unexpected_file.txt",
        ],
        "expected_success": False,
    },
])
@pytest.mark.data_read
def test_read_bad_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file and check problematic files (not quiet mode)
    data = srs.data.read(dataset, file_list, quiet=False)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)
    assert len(data.data) == 0

    # read file and check problematic files (quiet mode)
    data = srs.data.read(dataset, file_list, quiet=True)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)
    assert len(data.data) == 0

    # read file as single string input if only one file was supplied
    if (len(test_dict["filenames"]) == 0):
        # read file and check problematic files (not quiet mode)
        data = srs.data.read(dataset, test_dict["filenames"][0], quiet=False)
        assert len(data.problematic_files) > 0
        assert isinstance(data.problematic_files[0], ProblematicFile)
        assert len(data.data) == 0

        # read file and check problematic files (quiet mode)
        data = srs.data.read(dataset, test_dict["filenames"][0], quiet=True)
        assert len(data.problematic_files) > 0
        assert isinstance(data.problematic_files[0], ProblematicFile)
        assert len(data.data) == 0


@pytest.mark.data_read
def test_read_badperms_file(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # set filename
    f = "%s/20240205_mean-hsr_k0_v01.badperms.h5" % (DATA_DIR)
    os.chmod(f, 0o000)

    # read file and check problematic files (not quiet mode)
    data = srs.data.read(dataset, f, quiet=False)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file and check problematic files (quiet mode)
    data = srs.data.read(dataset, f, quiet=True)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # change perms back
    os.chmod(f, 0o644)


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 3,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_readers_func(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_swan_hsr(
        file_list,
        n_parallel=test_dict["n_parallel"],
        dataset=dataset,
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert len(data.data) == len(test_dict["filenames"])
    for item in data.data:
        assert item.raw_power.dtype == np.float32
        assert item.raw_power.shape[1] == item.timestamp.shape[0]
        assert len(item.raw_power) != 0
        assert len(item.timestamp) != 0
        assert len(item.band_central_frequency) > 0
        assert len(item.band_passband) > 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_readers_func_nodataset(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_swan_hsr(
        file_list,
        n_parallel=test_dict["n_parallel"],
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert len(data.data) == len(test_dict["filenames"])
    for item in data.data:
        assert item.raw_power.dtype == np.float32
        assert item.raw_power.shape[1] == item.timestamp.shape[0]
        assert len(item.raw_power) != 0
        assert len(item.timestamp) != 0
        assert len(item.band_central_frequency) > 0
        assert len(item.band_passband) > 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}

    # check __str__ and __repr__
    print_str = str(data)
    assert print_str != ""
    assert "dataset=None" in print_str


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "start_time": datetime.datetime(2024, 2, 3, 23, 50),
        "end_time": datetime.datetime(2024, 2, 5, 0, 5),
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "start_time": datetime.datetime(2024, 2, 3, 6, 12, 0),
        "end_time": datetime.datetime(2024, 2, 3, 6, 14, 0),
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "start_time": datetime.datetime(2024, 2, 3, 23, 55),
        "end_time": datetime.datetime(2024, 2, 4, 6, 30),
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "start_time": None,
        "end_time": datetime.datetime(2024, 2, 3, 6, 3, 30),
        "expected_success": True,
    },
    {
        "filenames": [
            "20240203_mean-hsr_k0_v01.h5",
            "20240204_mean-hsr_k0_v01.h5",
            "20240205_mean-hsr_k0_v01.h5",
        ],
        "start_time": datetime.datetime(2024, 2, 3, 6, 3, 30),
        "end_time": None,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_start_end_times(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SWAN_HSR_K0_H5")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    #
    # NOTE: we do this with a loop so we can do the same test for 1 and 2 values
    # for the n_parallel argument.
    for n_parallel in range(1, 2):
        data = srs.data.read(
            dataset,
            file_list,
            start_time=test_dict["start_time"],
            end_time=test_dict["end_time"],
            n_parallel=n_parallel,
        )

        # check success
        if (test_dict["expected_success"] is True):
            assert len(data.problematic_files) == 0
        else:
            assert len(data.problematic_files) > 0

        # check number of frames
        assert len(data.data) != 0
        for item in data.data:
            # base checks
            assert item.raw_power.dtype == np.float32
            assert item.raw_power.shape[1] == item.timestamp.shape[0]
            assert len(item.raw_power) != 0
            assert len(item.timestamp) != 0
            assert len(item.band_central_frequency) > 0
            assert len(item.band_passband) > 0

            # check that timestamps are in the valid range
            for t in item.timestamp:
                if (test_dict["start_time"] is not None):
                    assert t >= test_dict["start_time"]
                if (test_dict["end_time"] is not None):
                    assert t <= test_dict["end_time"]

        # check the metadata
        assert len(data.metadata) == len(data.data)
        assert len(data.metadata) != 0
        for m in data.metadata:
            assert m != {}
