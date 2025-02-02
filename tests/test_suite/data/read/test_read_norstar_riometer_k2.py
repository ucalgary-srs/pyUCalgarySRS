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
from pyucalgarysrs.data import ProblematicFile, RiometerData
from ...conftest import find_dataset

# globals
DATA_DIR = "%s/../../../test_data/read_norstar_riometer/k2" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "chu_rio_19930403_v1a.txt",
        "expected_success": True,
        "expected_frames": 17220
    },
    {
        "filename": "daw_rio_19930403_v1a.txt",
        "expected_success": True,
        "expected_frames": 17220
    },
    {
        "filename": "daw_rio_19980304_v1a.txt",
        "expected_success": True,
        "expected_frames": 17220
    },
    {
        "filename": "norstar_k2_rio-chur_20070402_v01.txt",
        "expected_success": True,
        "expected_frames": 17279
    },
    {
        "filename": "norstar_k2_rio-fsmi_20120504_v02.txt",
        "expected_success": True,
        "expected_frames": 17279
    },
    {
        "filename": "norstar_k2_rio-fsmi_20140202_v03.txt",
        "expected_success": True,
        "expected_frames": 17149
    },
    {
        "filename": "norstar_k2_rio-pina_20120504_v03.txt",
        "expected_success": True,
        "expected_frames": 16530
    },
    {
        "filename": "norstar_k2_rio-chur_20070402_v01.txt",
        "expected_success": True,
        "expected_frames": 17279
    },
    {
        "filename": "rab_rio_19900402_v1a.txt",
        "expected_success": True,
        "expected_frames": 17220
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_single_file(srs, all_datasets, test_dict, capsys):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

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
        assert isinstance(item, RiometerData) is True

    # check data
    assert data.data[0].raw_signal.shape == (test_dict["expected_frames"], )
    assert data.data[0].raw_signal.dtype == np.float32
    assert data.data[0].absorption.shape == (test_dict["expected_frames"], )
    assert data.data[0].absorption.dtype == np.float32
    assert data.data[0].timestamp.shape == (test_dict["expected_frames"], )
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

    # check __str__ and __repr__ for RiometerData type
    print_str = str(data.data[0])
    assert print_str != ""
    data.data[0].pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
        ],
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
        ],
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_multiple_files(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

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
        assert item.raw_signal.dtype == np.float32
        assert item.absorption.dtype == np.float32
        assert item.raw_signal.shape == item.timestamp.shape == item.absorption.shape
        assert len(item.raw_signal) != 0
        assert len(item.absorption) != 0
        assert len(item.timestamp) != 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "norstar_k2_rio-chur_20200501_v03.txt",
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filename": "norstar_k2_rio-chur_20200501_v03.txt",
        "n_parallel": 2,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_single_file_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

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
        assert item.raw_signal.dtype == np.float32
        assert item.absorption.dtype == np.float32
        assert item.raw_signal.shape == item.timestamp.shape == item.absorption.shape
        assert len(item.raw_signal) != 0
        assert len(item.absorption) != 0
        assert len(item.timestamp) != 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "n_parallel": 3,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_multiple_files_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

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
        assert item.raw_signal.dtype == np.float32
        assert item.absorption.dtype == np.float32
        assert item.raw_signal.shape == item.timestamp.shape == item.absorption.shape
        assert len(item.raw_signal) != 0
        assert len(item.absorption) != 0
        assert len(item.timestamp) != 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_first_record(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

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
            assert item.raw_signal.dtype == np.float32
            assert item.absorption.dtype == np.float32
            assert item.raw_signal.shape == item.timestamp.shape == item.absorption.shape
            assert len(item.raw_signal) != 0
            assert len(item.absorption) != 0
            assert len(item.timestamp) != 0

        # check the metadata
        assert len(data.metadata) == len(data.data)
        assert len(data.metadata) != 0
        for m in data.metadata:
            assert m != {}

        # check that the warning appeared
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "The 'first_record' parameter is not supported when reading NORSTAR riometer data." in str(w[-1].message)


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "n_parallel": 3,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_no_metadata(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

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
        assert item.raw_signal.dtype == np.float32
        assert item.absorption.dtype == np.float32
        assert item.raw_signal.shape == item.timestamp.shape == item.absorption.shape
        assert len(item.raw_signal) != 0
        assert len(item.absorption) != 0
        assert len(item.timestamp) != 0

    # check the metadata
    assert len(data.metadata) == 0


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "some_unexpected_file.txt",
        ],
        "expected_success": False,
    },
    {
        "filenames": [
            "mcm_rio_20030806_v1a.txt",
        ],
        "expected_success": False,
    },
    {
        "filenames": [
            "pin_rio_20050402_v1a.txt",
        ],
        "expected_success": False,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_bad_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

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
def test_read_norstar_riometer_k2_badperms_file(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

    # set filename
    f = "%s/rab_rio_19900402_v1a.badperms.txt" % (DATA_DIR)
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
            "norstar_k2_rio-chur_20200501_v03.txt",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "n_parallel": 3,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_readers_func(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_norstar_riometer(
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
        assert item.raw_signal.dtype == np.float32
        assert item.absorption.dtype == np.float32
        assert item.raw_signal.shape == item.timestamp.shape == item.absorption.shape
        assert len(item.raw_signal) != 0
        assert len(item.absorption) != 0
        assert len(item.timestamp) != 0

    # check the metadata
    assert len(data.metadata) == len(data.data)
    assert len(data.metadata) != 0
    for m in data.metadata:
        assert m != {}


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
        ],
        "n_parallel": 1,
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
        ],
        "n_parallel": 2,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_readers_func_nodataset(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_norstar_riometer(
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
        assert item.raw_signal.dtype == np.float32
        assert item.absorption.dtype == np.float32
        assert item.raw_signal.shape == item.timestamp.shape == item.absorption.shape
        assert len(item.raw_signal) != 0
        assert len(item.absorption) != 0
        assert len(item.timestamp) != 0

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
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "start_time": datetime.datetime(2020, 5, 1, 23, 50),
        "end_time": datetime.datetime(2020, 5, 3, 0, 5),
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "start_time": datetime.datetime(2020, 5, 1, 6, 12, 0),
        "end_time": datetime.datetime(2020, 5, 1, 6, 14, 0),
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "start_time": datetime.datetime(2020, 5, 1, 23, 55),
        "end_time": datetime.datetime(2020, 5, 2, 6, 30),
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "start_time": None,
        "end_time": datetime.datetime(2020, 5, 3, 6, 3, 30),
        "expected_success": True,
    },
    {
        "filenames": [
            "norstar_k2_rio-chur_20200501_v03.txt",
            "norstar_k2_rio-chur_20200502_v03.txt",
            "norstar_k2_rio-chur_20200503_v03.txt",
        ],
        "start_time": datetime.datetime(2020, 5, 3, 6, 3, 30),
        "end_time": None,
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_norstar_riometer_k2_start_end_times(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "NORSTAR_RIOMETER_K2_TXT")

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
            for item in data.data:
                assert item.raw_signal.dtype == np.float32
                assert item.absorption.dtype == np.float32
                assert item.raw_signal.shape == item.timestamp.shape == item.absorption.shape
                assert len(item.raw_signal) != 0
                assert len(item.absorption) != 0
                assert len(item.timestamp) != 0

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
