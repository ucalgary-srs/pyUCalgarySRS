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
DATA_DIR = "%s/../../../test_data/read_themis" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20140310_0600_gill_themis19_full.pgm.gz",
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20140310_0601_gill_themis19_full.pgm.gz",
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20140310_0605_gill_themis19_full.pgm",
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20040825_0526_atha_themis01_full_1000ms.pgm.bz2",
        "expected_success": True,
        "expected_frames": 10
    },
])
@pytest.mark.data_read
def test_read_themis_single_file(srs, all_datasets, test_dict, capsys):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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

    # check __str__ and __repr__ for Data type
    print_str = str(data)
    assert print_str != ""
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20140310_0605_gill_themis19_full.pgm",
        ],
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0605_gill_themis19_full.pgm",
        ],
        "expected_success": True,
        "expected_frames": 60
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_themis_multiple_files(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
    assert data.data.shape == (256, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20140310_0600_gill_themis19_full.pgm.gz",
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20140310_0601_gill_themis19_full.pgm.gz",
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 20
    },
])
@pytest.mark.data_read
def test_read_themis_single_file_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 100
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 100
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_themis_multiple_files_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0605_gill_themis19_full.pgm",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 3
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_themis_first_record(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0605_gill_themis19_full.pgm",
        ],
        "n_parallel": 3,
        "expected_success": True,
        "expected_frames": 60
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_themis_no_metadata(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_themis_first_record_and_no_metadata(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
            "20230104_0606_talo_themis15_full.pgm",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20210101_0332_gill_themis19_full.pgm",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20210101_0332_gill_themis19_full.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20211226_1624_gako_themis20_full.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20151204_0621_inuv_themis17_full.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20151030_0711_tpas_themis05_full.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20151106_0037_tpas_themis05_full.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20150729_0246_gbay_themis03_full.pgm.gz",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
])
@pytest.mark.data_read
def test_read_themis_bad_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
def test_read_themis_badperms_file(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

    # set filename
    f = "%s/20211226_1624_gako_themis20_full_badperms.pgm.gz" % (DATA_DIR)
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
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_themis_readers_func(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_themis(
        file_list,
        n_parallel=test_dict["n_parallel"],
        first_record=True,
        dataset=dataset,
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
            "20140310_0600_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
])
@pytest.mark.data_read
def test_read_themis_readers_func_nodataset(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_themis(
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
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "start_time": datetime.datetime(2014, 3, 10, 6, 1),
        "end_time": datetime.datetime(2014, 3, 10, 6, 3),
        "expected_success": True,
        "expected_frames": 41
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "start_time": datetime.datetime(2014, 3, 10, 6, 1, 15),
        "end_time": datetime.datetime(2014, 3, 10, 6, 3),
        "expected_success": True,
        "expected_frames": 36
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "start_time": datetime.datetime(2014, 3, 10, 6, 1),
        "end_time": datetime.datetime(2014, 3, 10, 6, 3, 15),
        "expected_success": True,
        "expected_frames": 46
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "start_time": None,
        "end_time": datetime.datetime(2014, 3, 10, 6, 3, 30),
        "expected_success": True,
        "expected_frames": 71
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "start_time": datetime.datetime(2014, 3, 10, 6, 3, 30),
        "end_time": None,
        "expected_success": True,
        "expected_frames": 30
    },
])
@pytest.mark.data_read
def test_read_themis_start_end_times(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "start_time": datetime.datetime(2014, 3, 10, 6, 3, 30),
        "end_time": None,
        "expected_success": True,
        "expected_frames": 100
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "start_time": None,
        "end_time": datetime.datetime(2014, 3, 10, 6, 3, 30),
        "expected_success": True,
        "expected_frames": 100
    },
    {
        "filenames": [
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
        ],
        "start_time": datetime.datetime(2014, 3, 10, 6, 3, 30),
        "end_time": datetime.datetime(2014, 3, 10, 6, 4, 30),
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_themis_nometa_startend(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")

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
