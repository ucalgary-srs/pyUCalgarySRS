import os
import pytest
import numpy as np
from pyucalgarysrs.data import ProblematicFile
from ...conftest import find_dataset

# globals
DATA_DIR = "%s/../../../test_data/read_trex_rgb/stream0.burst" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20211030_0600_gill_rgb-04_burst.png.tar",
        "expected_success": True,
        "expected_frames": 167
    },
    {
        "filename": "20211030_0601_gill_rgb-04_burst.png.tar",
        "expected_success": True,
        "expected_frames": 159
    },
    {
        "filename": "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
        "expected_success": True,
        "expected_frames": 1
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_burst_single_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

    # read file
    data = srs.data.read(dataset, "%s/%s" % (DATA_DIR, test_dict["filename"]))

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


@pytest.mark.data_read
def test_read_trex_rgb_burst_single_file_720p(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

    # read file
    data = srs.data.read(dataset, "%s/20181208_1308_fsmi_rgb-01_mode-b3_raw.png.tar" % (DATA_DIR))

    # check success
    assert len(data.problematic_files) == 0

    # check number of frames
    assert data.data.shape == (720, 830, 3, 33)
    assert len(data.metadata) == 33

    # check dtype
    assert data.data.dtype == np.uint8


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
        ],
        "expected_success": True,
        "expected_frames": 167
    },
    {
        "filenames": [
            "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
        ],
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
        ],
        "expected_success": True,
        "expected_frames": 167 + 159
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
            "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
        ],
        "expected_success": True,
        "expected_frames": 167 + 159 + 1
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_burst_multiple_files(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

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
    assert data.data.shape == (480, 553, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check dtype
    assert data.data.dtype == np.uint8


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20211030_0600_gill_rgb-04_burst.png.tar",
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 167
    },
    {
        "filename": "20211030_0601_gill_rgb-04_burst.png.tar",
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 159
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_burst_single_file_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

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
            "20211030_0600_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 167
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 167
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 167 + 159
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 167 + 159
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
            "20211030_0602_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 3,
        "expected_success": True,
        "expected_frames": 167 + 159 + 167
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_burst_multiple_files_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

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
            "20211030_0600_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
            "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
        ],
        "n_parallel": 3,
        "expected_success": True,
        "expected_frames": 3
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_burst_first_frame(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

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
            "20211030_0600_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 167
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 167 + 159
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
            "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
        ],
        "n_parallel": 3,
        "expected_success": True,
        "expected_frames": 167 + 159 + 1
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
            "20211030_0602_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 3,
        "expected_success": True,
        "expected_frames": 167 + 159 + 167
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_burst_no_metadata(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

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
            "20211030_0600_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
        ],
        "n_parallel": 1,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 2,
        "expected_success": True,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
            "20211030_0602_gill_rgb-04_burst.png.tar",
        ],
        "n_parallel": 3,
        "expected_success": True,
        "expected_frames": 3
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_burst_first_frame_and_no_metadata(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

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
            "20191121_0901_rabb_rgb-05_burst.png.tar",
        ],
        "expected_success": False,
        "expected_frames": 1
    },
])
@pytest.mark.data_read
def test_read_trex_rgb_burst_bad_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

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
def test_read_trex_rgb_burst_badperms_file(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "TREX_RGB_RAW_BURST")

    # set filename
    f = "%s/20211030_060500_149606_gill_rgb-04_320ms_burst_badperms.png" % (DATA_DIR)
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
