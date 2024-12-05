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
from pyucalgarysrs.exceptions import SRSUnsupportedReadError

# globals
DATA_DIR = "%s/../../../test_data/read_trex_spectrograph" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20230503_0600_luck_spect-02_spectra.pgm.gz",
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filename": "20230503_0601_luck_spect-02_spectra.pgm.gz",
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filename": "20230503_0605_luck_spect-02_spectra.pgm",
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filename": "20230101_0035_luck_spect-02_spectra_dark.pgm.gz",
        "expected_success": True,
        "expected_frames": 1
    },
])
@pytest.mark.data_read
def test_read_trex_spectrograph_single_file(srs, test_dict):
    # read file
    data = srs.data.readers.read_trex_spectrograph("%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (1024, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filenames": [
            "20230503_0605_luck_spect-02_spectra.pgm",
        ],
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 8
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0605_luck_spect-02_spectra.pgm",
        ],
        "expected_success": True,
        "expected_frames": 12
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 20
    },
])
@pytest.mark.data_read
def test_read_trex_spectrograph_multiple_files(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_spectrograph(file_list)

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (1024, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20230503_0600_luck_spect-02_spectra.pgm.gz",
        "workers": 1,
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filename": "20230503_0601_luck_spect-02_spectra.pgm.gz",
        "workers": 2,
        "expected_success": True,
        "expected_frames": 4
    },
])
@pytest.mark.data_read
def test_read_trex_spectrograph_single_file_workers(srs, test_dict):
    # read file
    data = srs.data.readers.read_trex_spectrograph(
        "%s/%s" % (DATA_DIR, test_dict["filename"]),
        n_parallel=test_dict["workers"],
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (1024, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 8
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 8
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 5,
        "expected_success": True,
        "expected_frames": 20
    },
])
@pytest.mark.data_read
def test_read_trex_spectrograph_multiple_files_workers(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_spectrograph(
        file_list,
        n_parallel=test_dict["workers"],
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (1024, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0605_luck_spect-02_spectra.pgm",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 3
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_spectrograph_first_frame(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_spectrograph(
        file_list,
        n_parallel=test_dict["workers"],
        first_record=True,
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (1024, 256, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 4
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 8
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0605_luck_spect-02_spectra.pgm",
        ],
        "workers": 3,
        "expected_success": True,
        "expected_frames": 12
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 5,
        "expected_success": True,
        "expected_frames": 20
    },
])
@pytest.mark.data_read
def test_read_trex_spectrograph_no_metadata(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_spectrograph(
        file_list,
        n_parallel=test_dict["workers"],
        no_metadata=True,
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (1024, 256, test_dict["expected_frames"])
    assert len(data.metadata) == 0

    # check that there's no metadata
    for m in data.metadata:
        assert len(m) == 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
        ],
        "workers": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_spectrograph_first_frame_and_no_metadata(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_spectrograph(
        file_list,
        n_parallel=test_dict["workers"],
        first_record=True,
        no_metadata=True,
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (1024, 256, test_dict["expected_frames"])
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
        "exception": SRSUnsupportedReadError,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20190930_0559_luck_spect-02_spectra.pgm.gz",
        ],
        "expected_success": False,
        "exception": None,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20210909_0322_rabb_spect-01_spectra.pgm.gz",
        ],
        "expected_success": False,
        "exception": None,
        "expected_frames": 1
    },
])
@pytest.mark.data_read
def test_read_trex_spectrograph_bad_file(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file and check problematic files (not quiet mode)
    try:
        data = srs.data.readers.read_trex_spectrograph(file_list, quiet=False)
    except SRSUnsupportedReadError as e:
        if (test_dict["exception"] is not None and test_dict["exception"] is SRSUnsupportedReadError):
            assert True
            return
        else:
            raise AssertionError("SRSUnsupportedReadError occurred when not expected") from e

    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file and check problematic files (quiet mode)
    data = srs.data.readers.read_trex_spectrograph(file_list, quiet=True)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file as single string input if only one file was supplied
    if (len(test_dict["filenames"]) == 0):
        # read file and check problematic files (not quiet mode)
        data = srs.data.readers.read_trex_spectrograph(test_dict["filenames"][0], quiet=False)
        assert len(data.problematic_files) > 0
        assert isinstance(data.problematic_files[0], ProblematicFile)

        # read file and check problematic files (quiet mode)
        data = srs.data.readers.read_trex_spectrograph(test_dict["filenames"][0], quiet=True)
        assert len(data.problematic_files) > 0
        assert isinstance(data.problematic_files[0], ProblematicFile)


@pytest.mark.data_read
def test_read_trex_spectrograph_badperms_file(srs):
    # set filename
    f = "%s/20230101_0600_rabb_spect-01_spectra.pgm.gz.badperms" % (DATA_DIR)
    os.chmod(f, 0o000)

    # read file and check problematic files (not quiet mode)
    data = srs.data.readers.read_trex_spectrograph(f, quiet=False)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # read file and check problematic files (quiet mode)
    data = srs.data.readers.read_trex_spectrograph(f, quiet=True)
    assert len(data.problematic_files) > 0
    assert isinstance(data.problematic_files[0], ProblematicFile)

    # change perms back
    os.chmod(f, 0o644)
