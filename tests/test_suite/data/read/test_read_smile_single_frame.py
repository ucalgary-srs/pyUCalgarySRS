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
#
# NOTE: these tests cover SMILE ASI raw files that contain a single image frame,
# instead of the usual minute of frames. These files show up when the bundling of
# individual frames into a minute-long file didn't happen onsite. They differ from
# regular files in two ways:
#
#   1) the 'data/images' dataset is 3-dimensional (no frame dimension), instead of
#      being 4-dimensional
#   2) the metadata attributes are stored as single-element arrays, instead of as
#      scalar values

import os
import datetime
import pytest
import numpy as np
from pathlib import Path
from ...conftest import find_dataset

# globals
DATA_DIR = "%s/../../../test_data/read_smile/single_frame" % (os.path.dirname(os.path.realpath(__file__)))
BUNDLED_DATA_DIR = "%s/../../../test_data/read_smile" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "20250921_020000_000053_pina_smile-05_raw.h5",
        "expected_timestamp": datetime.datetime(2025, 9, 21, 2, 0, 0, 53),
    },
    {
        "filename": "20250921_020003_000076_pina_smile-05_raw.h5",
        "expected_timestamp": datetime.datetime(2025, 9, 21, 2, 0, 3, 76),
    },
    {
        "filename": "20250921_020006_000053_pina_smile-05_raw.h5",
        "expected_timestamp": datetime.datetime(2025, 9, 21, 2, 0, 6, 53),
    },
])
@pytest.mark.data_read
def test_read_single_frame_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SMILE_ASI_RAW")

    # read file
    data = srs.data.read(dataset, "%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check success
    assert len(data.problematic_files) == 0

    # check that the single frame was reshaped up to 4 dimensions
    assert data.data.shape == (512, 512, 3, 1)
    assert data.data.dtype == np.uint8

    # check metadata and timestamp
    assert len(data.metadata) == 1
    assert len(data.metadata[0]) > 0
    assert len(data.timestamp) == 1
    assert data.timestamp[0] == test_dict["expected_timestamp"]


@pytest.mark.data_read
def test_read_single_frame_metadata_values_are_scalars(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "SMILE_ASI_RAW")

    # read file
    #
    # NOTE: the metadata attributes in these files are stored as single-element arrays,
    # so we check that they get unwrapped into scalar values. Without this, deriving the
    # timestamp from the metadata fails.
    data = srs.data.read(dataset, "%s/20250921_020000_000053_pina_smile-05_raw.h5" % (DATA_DIR))

    # check success
    assert len(data.problematic_files) == 0

    # check that no metadata value is an array
    assert len(data.metadata) == 1
    for key, value in data.metadata[0].items():
        assert not isinstance(value, np.ndarray), "metadata key '%s' was not unwrapped to a scalar" % (key)
        assert isinstance(value, str)

    # check some specific values
    assert data.metadata[0]["image_request_start"] == "2025-09-21 02:00:00.000053 UTC"
    assert data.metadata[0]["global_gain"] == "11.0000 dB"


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "20250921_020000_000053_pina_smile-05_raw.h5",
        ],
        "n_parallel": 1,
        "expected_frames": 1
    },
    {
        "filenames": [
            "20250921_020000_000053_pina_smile-05_raw.h5",
            "20250921_020003_000076_pina_smile-05_raw.h5",
        ],
        "n_parallel": 1,
        "expected_frames": 2
    },
    {
        "filenames": [
            "20250921_020000_000053_pina_smile-05_raw.h5",
            "20250921_020003_000076_pina_smile-05_raw.h5",
            "20250921_020006_000053_pina_smile-05_raw.h5",
        ],
        "n_parallel": 1,
        "expected_frames": 3
    },
    {
        "filenames": [
            "20250921_020000_000053_pina_smile-05_raw.h5",
            "20250921_020003_000076_pina_smile-05_raw.h5",
            "20250921_020006_000053_pina_smile-05_raw.h5",
        ],
        "n_parallel": 2,
        "expected_frames": 3
    },
    {
        "filenames": [
            "20250921_020000_000053_pina_smile-05_raw.h5",
            "20250921_020003_000076_pina_smile-05_raw.h5",
            "20250921_020006_000053_pina_smile-05_raw.h5",
        ],
        "n_parallel": 3,
        "expected_frames": 3
    },
])
@pytest.mark.data_read
def test_read_multiple_single_frame_files(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SMILE_ASI_RAW")

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read files
    data = srs.data.read(dataset, file_list, n_parallel=test_dict["n_parallel"])

    # check success
    assert len(data.problematic_files) == 0

    # check number of frames
    assert data.data.shape == (512, 512, 3, test_dict["expected_frames"])
    assert data.data.dtype == np.uint8
    assert len(data.metadata) == test_dict["expected_frames"]
    assert len(data.timestamp) == test_dict["expected_frames"]

    # check that there's metadata
    for m in data.metadata:
        assert len(m) > 0

    # check that the timestamps are in ascending order
    assert data.timestamp == sorted(data.timestamp)


@pytest.mark.data_read
def test_read_single_frame_and_bundled_files_together(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "SMILE_ASI_RAW")

    # build file list of both flavours of file
    #
    # NOTE: a bundled file has 20 frames in it, and each single frame file has 1
    file_list = [
        "%s/20250315_0600_atha_smile-31_rgb-full.h5" % (BUNDLED_DATA_DIR),
        "%s/20250921_020000_000053_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250315_0601_atha_smile-31_rgb-full.h5" % (BUNDLED_DATA_DIR),
        "%s/20250921_020003_000076_pina_smile-05_raw.h5" % (DATA_DIR),
    ]

    # read files
    data = srs.data.read(dataset, file_list)

    # check success
    assert len(data.problematic_files) == 0

    # check number of frames
    assert data.data.shape == (512, 512, 3, 42)
    assert data.data.dtype == np.uint8
    assert len(data.metadata) == 42
    assert len(data.timestamp) == 42

    # check that all metadata values came out as scalars, no matter the flavour of file
    for m in data.metadata:
        assert len(m) > 0
        for value in m.values():
            assert not isinstance(value, np.ndarray)


@pytest.mark.parametrize("test_dict", [
    {
        "n_parallel": 1,
        "expected_frames": 3
    },
    {
        "n_parallel": 3,
        "expected_frames": 3
    },
])
@pytest.mark.data_read
def test_read_single_frame_first_record(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SMILE_ASI_RAW")

    # build file list
    #
    # NOTE: each file only has one frame in it, so the first record of each file is
    # the only record of each file
    file_list = [
        "%s/20250921_020000_000053_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250921_020003_000076_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250921_020006_000053_pina_smile-05_raw.h5" % (DATA_DIR),
    ]

    # read files
    data = srs.data.read(dataset, file_list, n_parallel=test_dict["n_parallel"], first_record=True)

    # check success
    assert len(data.problematic_files) == 0

    # check number of frames
    assert data.data.shape == (512, 512, 3, test_dict["expected_frames"])
    assert data.data.dtype == np.uint8
    assert len(data.metadata) == test_dict["expected_frames"]


@pytest.mark.data_read
def test_read_single_frame_no_metadata(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "SMILE_ASI_RAW")

    # build file list
    file_list = [
        "%s/20250921_020000_000053_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250921_020003_000076_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250921_020006_000053_pina_smile-05_raw.h5" % (DATA_DIR),
    ]

    # read files
    data = srs.data.read(dataset, file_list, no_metadata=True)

    # check success
    assert len(data.problematic_files) == 0

    # check number of frames, and that no metadata came back
    assert data.data.shape == (512, 512, 3, 3)
    assert data.data.dtype == np.uint8
    assert len(data.metadata) == 0
    assert len(data.timestamp) == 0


@pytest.mark.parametrize("test_dict", [
    {
        "start_time": datetime.datetime(2025, 9, 21, 2, 0, 3),
        "end_time": None,
        "expected_frames": 2
    },
    {
        "start_time": None,
        "end_time": datetime.datetime(2025, 9, 21, 2, 0, 3),
        "expected_frames": 2
    },
    {
        "start_time": datetime.datetime(2025, 9, 21, 2, 0, 3),
        "end_time": datetime.datetime(2025, 9, 21, 2, 0, 3),
        "expected_frames": 1
    },
    {
        "start_time": datetime.datetime(2025, 9, 21, 2, 0, 0),
        "end_time": datetime.datetime(2025, 9, 21, 2, 0, 6),
        "expected_frames": 3
    },
])
@pytest.mark.data_read
def test_read_single_frame_start_end_times(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, "SMILE_ASI_RAW")

    # build file list
    file_list = [
        "%s/20250921_020000_000053_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250921_020003_000076_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250921_020006_000053_pina_smile-05_raw.h5" % (DATA_DIR),
    ]

    # read files
    data = srs.data.read(
        dataset,
        file_list,
        start_time=test_dict["start_time"],
        end_time=test_dict["end_time"],
    )

    # check success
    assert len(data.problematic_files) == 0

    # check number of frames
    assert data.data.shape == (512, 512, 3, test_dict["expected_frames"])
    assert len(data.metadata) == test_dict["expected_frames"]
    assert len(data.timestamp) == test_dict["expected_frames"]

    # check that timestamps are in the valid range
    for t in data.timestamp:
        t = t.replace(microsecond=0)
        if (test_dict["start_time"] is not None):
            assert t >= test_dict["start_time"]
        if (test_dict["end_time"] is not None):
            assert t <= test_dict["end_time"]


@pytest.mark.data_read
def test_read_single_frame_pathlib_input(srs, all_datasets):
    # set dataset
    dataset = find_dataset(all_datasets, "SMILE_ASI_RAW")

    # build file list
    file_list = [
        Path(DATA_DIR) / Path("20250921_020000_000053_pina_smile-05_raw.h5"),
        Path(DATA_DIR) / Path("20250921_020003_000076_pina_smile-05_raw.h5"),
    ]

    # read files
    data = srs.data.read(dataset, file_list)

    # check success
    assert len(data.problematic_files) == 0

    # check number of frames
    assert data.data.shape == (512, 512, 3, 2)
    assert len(data.metadata) == 2


@pytest.mark.data_read
def test_read_single_frame_readers_func(srs):
    # read files using the reader function directly, with no dataset supplied
    file_list = [
        "%s/20250921_020000_000053_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250921_020003_000076_pina_smile-05_raw.h5" % (DATA_DIR),
        "%s/20250921_020006_000053_pina_smile-05_raw.h5" % (DATA_DIR),
    ]
    data = srs.data.readers.read_smile(file_list)

    # check success
    assert len(data.problematic_files) == 0

    # check number of frames
    assert data.data.shape == (512, 512, 3, 3)
    assert data.data.dtype == np.uint8
    assert len(data.metadata) == 3
    assert len(data.timestamp) == 3
