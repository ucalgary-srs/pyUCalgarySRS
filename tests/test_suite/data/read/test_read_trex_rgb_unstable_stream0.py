import os
import pytest
import numpy as np

# globals
DATA_DIR = "%s/../../../test_data/read_trex_rgb/unstable/stream0" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.data_read
@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20210503_0600_luck_rgb-03_full.pgm.gz",
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20210503_0601_luck_rgb-03_full.pgm.gz",
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20210503_0605_luck_rgb-03_full.pgm",
        "expected_success": True,
        "expected_frames": 20
    },
])
def test_read_trex_rgb_unstable_stream0_single_file(srs, test_dict):
    # read file
    data = srs.data.readers.read_trex_rgb("%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (480, 553, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20210503_0605_luck_rgb-03_full.pgm",
        ],
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0605_luck_rgb-03_full.pgm",
        ],
        "expected_success": True,
        "expected_frames": 60
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0_multiple_files(srs, test_dict):
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
    assert data.data.shape == (480, 553, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20210503_0600_luck_rgb-03_full.pgm.gz",
        "workers": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filename": "20210503_0601_luck_rgb-03_full.pgm.gz",
        "workers": 2,
        "expected_success": True,
        "expected_frames": 20
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0_single_file_workers(srs, test_dict):
    # read file
    data = srs.data.readers.read_trex_rgb(
        "%s/%s" % (DATA_DIR, test_dict["filename"]),
        n_parallel=test_dict["workers"],
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (480, 553, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 100
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 100
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 5,
        "expected_success": True,
        "expected_frames": 100
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0_multiple_files_workers(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(
        file_list,
        n_parallel=test_dict["workers"],
    )

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == (480, 553, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0605_luck_rgb-03_full.pgm",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 3
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 5
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0_first_frame(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(
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
    assert data.data.shape == (480, 553, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.data_read
@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 20
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 40
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0605_luck_rgb-03_full.pgm",
        ],
        "workers": 3,
        "expected_success": True,
        "expected_frames": 60
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 5,
        "expected_success": True,
        "expected_frames": 100
    },
])
def test_read_trex_rgb_unstable_stream0_no_metadata(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(
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
    assert data.data.shape == (480, 553, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's no metadata
    for m in data.metadata:
        assert len(m) == 0

    # check dtype
    assert data.data.dtype == np.uint16


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20210503_0600_luck_rgb-03_full.pgm.gz",
            "20210503_0601_luck_rgb-03_full.pgm.gz",
            "20210503_0602_luck_rgb-03_full.pgm.gz",
            "20210503_0603_luck_rgb-03_full.pgm.gz",
            "20210503_0604_luck_rgb-03_full.pgm.gz",
        ],
        "workers": 5,
        "expected_success": True,
        "expected_frames": 5
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_unstable_stream0_first_frame_and_no_metadata(srs, test_dict):
    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read file
    data = srs.data.readers.read_trex_rgb(
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
    assert data.data.shape == (480, 553, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's no metadata
    for m in data.metadata:
        assert len(m) == 0

    # check dtype
    assert data.data.dtype == np.uint16
