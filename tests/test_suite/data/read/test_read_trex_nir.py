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
from pyucalgarysrs.data import ProblematicFile
from ...conftest import find_dataset

# globals
DATA_DIR = "%s/../../../test_data/read_trex_nir" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20220307_0600_gill_nir-216_8446.pgm.gz",
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filename": "20220307_0601_gill_nir-216_8446.pgm.gz",
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filename": "20220307_0605_gill_nir-216_8446.pgm",
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filename": "20220101_2329_gill_nir-216_8446_dark.pgm.gz",
        "expected_success": True,
        "expected_frames": 3
    },
])
@pytest.mark.data_read
def test_read_trex_nir_single_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

    # read file
    data = srs.data.read(dataset, "%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filenames": [
            "20220307_0605_gill_nir-216_8446.pgm",
        ],
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0605_gill_nir-216_8446.pgm",
        ],
        "expected_success": True,
        "expected_frames": 30
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 50
    },
])
@pytest.mark.data_read
def test_read_trex_nir_multiple_files(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.read(dataset, file_list)

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20220307_0600_gill_nir-216_8446.pgm.gz",
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filename": "20220307_0601_gill_nir-216_8446.pgm.gz",
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 10
    },
])
@pytest.mark.data_read
def test_read_trex_nir_single_file_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

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
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 50
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 50
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 50
    },
])
@pytest.mark.data_read
def test_read_trex_nir_multiple_files_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

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

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0605_gill_nir-216_8446.pgm",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 3
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_nir_first_frame(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

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
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 10
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0605_gill_nir-216_8446.pgm",
        ],
        "n_parallel": 3,
        "expected_success": True,
        "expected_frames": 30
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 50
    },
])
@pytest.mark.data_read
def test_read_trex_nir_no_metadata(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

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
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == 0

    # check that there's no metadata
    for m in data.metadata:
        assert len(m) == 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_nir_first_frame_and_no_metadata(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.read(
        dataset,
        file_list,
        n_parallel=test_dict["n_parallel"],
        first_record=True,
        no_metadata=True,
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == 0

    # check that there's no metadata
    for m in data.metadata:
        assert len(m) == 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "some_unexpected_file.txt",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20220210_1043_atha_nir-221_8446.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20221004_1002_luck_nir-219_8446.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20191126_0600_gill_nir-217_8446.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
])
@pytest.mark.data_read
def test_read_trex_nir_bad_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file and check problematic files (not quiet mode)
    data = srs.data.read(dataset, file_list, quiet=False)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file and check problematic files (quiet mode)
    data = srs.data.read(dataset, file_list, quiet=True)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file as single string input if only one file was supplied
    if (len(test_dict["filenames"]) == 0):
        # read file and check problematic files (not quiet mode)
        data = srs.data.read(dataset, test_dict["filenames"][0], quiet=False)
        assert len(data.problematic_files) > 0
        assert isinstance(data.problematic_files[0], ProblematicFile)

        # read file and check problematic files (quiet mode)
        data = srs.data.read(dataset, test_dict["filenames"][0], quiet=True)
        assert len(data.problematic_files) > 0
        assert isinstance(data.problematic_files[0], ProblematicFile)


@pytest.mark.data_read
def test_read_trex_nir_badperms_file(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

    # set filename
    f = "%s/20200101_0700_gill_nir-217_8446_badperms.pgm.gz" % (DATA_DIR)
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
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_nir_readers_func(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_nir(
        file_list,
        n_parallel=test_dict["n_parallel"],
        dataset=dataset,
        first_record=True,
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
])
@pytest.mark.data_read
def test_read_trex_nir_readers_func_nodataset(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_nir(
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
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check dtype
    assert data.data.dtype == np.uint16

    # check __str__ and __repr__
    print_str = str(data)
    assert print_str != ""
    assert "dataset=None" in print_str


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "start_time": datetime.datetime(2022, 3, 7, 6, 1),
        "end_time": datetime.datetime(2022, 3, 7, 6, 3),
        "expected_success": True,
        "expected_frames": 21
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "start_time": datetime.datetime(2022, 3, 7, 6, 1, 15),
        "end_time": datetime.datetime(2022, 3, 7, 6, 3),
        "expected_success": True,
        "expected_frames": 18
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "start_time": datetime.datetime(2022, 3, 7, 6, 1),
        "end_time": datetime.datetime(2022, 3, 7, 6, 3, 15),
        "expected_success": True,
        "expected_frames": 23
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "start_time": None,
        "end_time": datetime.datetime(2022, 3, 7, 6, 3, 30),
        "expected_success": True,
        "expected_frames": 36
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "start_time": datetime.datetime(2022, 3, 7, 6, 3, 30),
        "end_time": None,
        "expected_success": True,
        "expected_frames": 15
    },
])
@pytest.mark.data_read
def test_read_trex_nir_start_end_times(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

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
        assert data.data.shape == (256, 256, test_dict["expected_frames"])
        assert len(data.metadata) == test_dict["expected_frames"]
        assert len(data.timestamp) == test_dict["expected_frames"]

        # check that there's metadata
        for m in data.metadata:
            assert len(m) > 0

        # check that timestamps are in the valid range
        for t in data.timestamp:
            t = t.replace(microsecond=0)
            if (test_dict["start_time"] is not None):
                assert t >= test_dict["start_time"]
            if (test_dict["end_time"] is not None):
                assert t <= test_dict["end_time"]


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "start_time": datetime.datetime(2022, 3, 7, 6, 3, 30),
        "end_time": None,
        "expected_success": True,
        "expected_frames": 50
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "start_time": None,
        "end_time": datetime.datetime(2022, 3, 7, 6, 3, 30),
        "expected_success": True,
        "expected_frames": 50
    },
    {
        "filenames": [
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
        ],
        "start_time": datetime.datetime(2022, 3, 7, 6, 3, 30),
        "end_time": datetime.datetime(2022, 3, 7, 6, 4, 30),
        "expected_success": True,
        "expected_frames": 50
    },
])
@pytest.mark.data_read
def test_read_trex_nir_nometa_startend(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_NIR_RAW")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    with warnings.catch_warnings(record=True) as w:
        data = srs.data.read(
            dataset,
            file_list,
            start_time=test_dict["start_time"],
            end_time=test_dict["end_time"],
            no_metadata=True,
        )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == 0
    assert len(data.timestamp) == 0

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check that the warning appeared
    assert len(w) == 1
    assert issubclass(w[-1].category, UserWarning)
    assert "Cannot filter on start or end time if the no_metadata parameter is set to True" in str(w[-1].message)
