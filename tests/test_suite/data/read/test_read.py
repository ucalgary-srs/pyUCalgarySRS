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
from pyucalgarysrs import SRSUnsupportedReadError
from ...conftest import find_dataset


@pytest.mark.parametrize("test_dict", [
    {
        "dataset_name": "THEMIS_ASI_RAW",
        "filename": "%s/../../../test_data/read_themis/20140310_0600_gill_themis19_full.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
        "expected_success": True,
        "expected_frames": 20,
        "expected_dims": (256, 256, 20),
        "expected_dtype": np.uint16,
    },
    {
        "dataset_name": "THEMIS_ASI_RAW",
        "filename": [
            "%s/../../../test_data/read_themis/20140310_0600_gill_themis19_full.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
            "%s/../../../test_data/read_themis/20140310_0601_gill_themis19_full.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
        ],
        "expected_success": True,
        "expected_frames": 40,
        "expected_dims": (256, 256, 40),
        "expected_dtype": np.uint16,
    },
    {
        "dataset_name": "REGO_RAW",
        "filename": "%s/../../../test_data/read_rego/20180403_0600_gill_rego-652_6300.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
        "expected_success": True,
        "expected_frames": 20,
        "expected_dims": (512, 512, 20),
        "expected_dtype": np.uint16,
    },
    {
        "dataset_name": "REGO_RAW",
        "filename": [
            "%s/../../../test_data/read_rego/20180403_0600_gill_rego-652_6300.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
            "%s/../../../test_data/read_rego/20180403_0601_gill_rego-652_6300.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
        ],
        "expected_success": True,
        "expected_frames": 40,
        "expected_dims": (512, 512, 40),
        "expected_dtype": np.uint16,
    },
    {
        "dataset_name": "TREX_NIR_RAW",
        "filename": "%s/../../../test_data/read_trex_nir/20220307_0600_gill_nir-216_8446.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
        "expected_success": True,
        "expected_frames": 10,
        "expected_dims": (256, 256, 10),
        "expected_dtype": np.uint16,
    },
    {
        "dataset_name": "TREX_NIR_RAW",
        "filename": [
            "%s/../../../test_data/read_trex_nir/20220307_0600_gill_nir-216_8446.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
            "%s/../../../test_data/read_trex_nir/20220307_0601_gill_nir-216_8446.pgm.gz" % (os.path.dirname(os.path.realpath(__file__))),
        ],
        "expected_success": True,
        "expected_frames": 20,
        "expected_dims": (256, 256, 20),
        "expected_dtype": np.uint16,
    },
])
@pytest.mark.data_read
def test_read_generic(srs, capsys, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # read file
    data = srs.data.read(dataset, test_dict["filename"])

    # check success
    if (test_dict["expected_success"] is True):
        assert len(data.problematic_files) == 0
    else:
        assert len(data.problematic_files) > 0

    # check number of frames
    assert data.data.shape == test_dict["expected_dims"]
    assert len(data.metadata) == test_dict["expected_frames"]

    # check dtype
    assert data.data.dtype == np.uint16

    # check __str__ and __repr__
    print_str = str(data)
    assert print_str != ""

    # check pretty print method
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.data_read
def test_read_generic_bad_dataset(srs, all_datasets):
    # set dataset by hacking it a bit
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")
    if (dataset is None):
        return
    dataset.name = "bad_dataset_name"

    # check that the read routine raises an exception
    with pytest.raises(SRSUnsupportedReadError) as e_info:
        _ = srs.data.read(dataset, "some_fake_filename.pgm")
    assert "Dataset does not have a supported read function" in str(e_info)


@pytest.mark.data_read
def test_read_generic_none_dataset(srs):
    # check that the read routine raises an exception
    with pytest.raises(SRSUnsupportedReadError) as e_info:
        _ = srs.data.read(None, "some_fake_filename.pgm")
    assert "Must supply a dataset" in str(e_info)


@pytest.mark.data_read
def test_list_supported_read_datasets(srs):
    datasets = srs.data.list_supported_read_datasets()
    assert len(datasets) > 0


@pytest.mark.parametrize("test_dict", [
    {
        "dataset_name": "THEMIS_ASI_RAW",
        "expected_success": True,
    },
    {
        "dataset_name": "SOME_BAD_DATASET",
        "expected_success": False,
    },
])
@pytest.mark.data_read
def test_is_read_supported(srs, test_dict):
    assert srs.data.is_read_supported(test_dict["dataset_name"]) is test_dict["expected_success"]


@pytest.mark.parametrize("test_dict", [
    {
        "dataset_name": "THEMIS_ASI_RAW",
        "filename": [],
        "expected_success": True,
    },
    {
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
        "filename": [],
        "expected_success": True,
    },
    {
        "dataset_name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
        "filename": [],
        "expected_success": True,
    },
])
@pytest.mark.data_read
def test_read_no_files(srs, all_datasets, test_dict, capsys):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # read file
    data = srs.data.read(dataset, test_dict["filename"])

    # check data size
    assert len(data.timestamp) == 0
    if (isinstance(data.data, np.ndarray) is True):
        assert data.data.shape[-1] == 0
    elif (isinstance(data.data, list) is True):
        assert len(data.data) == 0
    else:
        raise AssertionError("Error in returned data type; not ndarray or list, but is %s instead" % (type(data.data)))

    # check __str__ and __repr__
    print_str = str(data)
    assert print_str != ""

    # check pretty print method
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""
