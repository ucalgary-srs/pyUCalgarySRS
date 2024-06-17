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
import pytest
import numpy as np
from pyucalgarysrs.data import ProblematicFile

# globals
DATA_DIR = "%s/../../../test_data/read_trex_rgb/unstable/stream0.colour" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20200508_0600_gill_rgb-04_full.png.tar",
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20200508_0601_gill_rgb-04_full.png.tar",
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20200508_060500_122643_gill_rgb-04_320ms_full.png",
        "expected_success": True,
        "expected_frames": 1
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_single_file(srs, test_dict):
    # read file
    data = srs.data.readers.read_trex_rgb("%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (480, 553, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check dtype
    assert data.data.dtype == np.uint8


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
        ],
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20200508_060500_122643_gill_rgb-04_320ms_full.png",
        ],
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
        ],
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_060500_122643_gill_rgb-04_320ms_full.png",
        ],
        "expected_success": True,
        "expected_frames": 41
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_multiple_files(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(file_list)

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (480, 553, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint8


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20200508_0600_gill_rgb-04_full.png.tar",
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20200508_0601_gill_rgb-04_full.png.tar",
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 20
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_single_file_n_parallel(srs, test_dict):
    # read file
    data = srs.data.readers.read_trex_rgb(
        "%s/%s" % (DATA_DIR, test_dict["filename"]),
        n_parallel=test_dict["n_parallel"],
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (480, 553, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint8


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 100
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 100
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_multiple_files_n_parallel(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(
        file_list,
        n_parallel=test_dict["n_parallel"],
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (480, 553, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint8


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_060500_122643_gill_rgb-04_320ms_full.png",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 3
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_first_frame(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(
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
    assert data.data.shape == (480, 553, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint8


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_060500_122643_gill_rgb-04_320ms_full.png",
        ],
        "n_parallel": 3,
        "expected_success": True,
        "expected_frames": 41
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_no_metadata(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(
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
    assert data.data.shape == (480, 553, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's no metadata
    for m in data.metadata:
        assert len(m) == 0

    # check dtype
    assert data.data.dtype == np.uint8


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_060500_122643_gill_rgb-04_320ms_full.png",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_060500_122643_gill_rgb-04_320ms_full.png",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20200508_0600_gill_rgb-04_full.png.tar",
            "20200508_0601_gill_rgb-04_full.png.tar",
            "20200508_0602_gill_rgb-04_full.png.tar",
            "20200508_0603_gill_rgb-04_full.png.tar",
            "20200508_0604_gill_rgb-04_full.png.tar",
        ],
        "n_parallel": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_first_frame_and_no_metadata(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(
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
    assert data.data.shape == (480, 553, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's no metadata
    for m in data.metadata:
        assert len(m) == 0

    # check dtype
    assert data.data.dtype == np.uint8


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
            "20220512_0711_fsmi_rgb-01_full.png.tar",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_bad_file(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file and check problematic files (not quiet mode)
    data = srs.data.readers.read_trex_rgb(file_list, quiet=False)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file and check problematic files (quiet mode)
    data = srs.data.readers.read_trex_rgb(file_list, quiet=True)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file as single string input if only one file was supplied
    if (len(test_dict["filenames"]) == 0):
        # read file and check problematic files (not quiet mode)
        data = srs.data.readers.read_trex_rgb(test_dict["filenames"][0], quiet=False)
        assert len(data.problematic_files) > 0
        assert isinstance(data.problematic_files[0], ProblematicFile)

        # read file and check problematic files (quiet mode)
        data = srs.data.readers.read_trex_rgb(test_dict["filenames"][0], quiet=True)
        assert len(data.problematic_files) > 0
        assert isinstance(data.problematic_files[0], ProblematicFile)


@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0colour_badperms_file(srs):
    # set filename
    f = "%s/20220512_0700_fsmi_rgb-01_full_badperms.png.tar" % (DATA_DIR)
    os.chmod(f, 0o000)

    # read file and check problematic files (not quiet mode)
    data = srs.data.readers.read_trex_rgb(f, quiet=False)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file and check problematic files (quiet mode)
    data = srs.data.readers.read_trex_rgb(f, quiet=True)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # change perms back
    os.chmod(f, 0o644)
