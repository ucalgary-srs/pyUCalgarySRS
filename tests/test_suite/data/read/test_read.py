import os
import pytest
import numpy as np
from pyucalgarysrs import SRSUnsupportedReadException
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
def test_read_generic(srs, all_datasets, test_dict):
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


@pytest.mark.data_read
def test_read_generic_bad_dataset(srs, all_datasets):
    # set dataset by hacking it a bit
    dataset = find_dataset(all_datasets, "THEMIS_ASI_RAW")
    if (dataset is None):
        return
    dataset.name = "bad_dataset_name"

    # check that the read routine raises an exception
    with pytest.raises(SRSUnsupportedReadException) as e_info:
        _ = srs.data.read(dataset, "some_fake_filename.pgm")
        assert "Dataset does not have a supported read function" == str(e_info)


@pytest.mark.data_read
def test_read_generic_none_dataset(srs):
    # check that the read routine raises an exception
    with pytest.raises(SRSUnsupportedReadException) as e_info:
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
def test_check_if_read_supported(srs, test_dict):
    assert srs.data.check_if_read_supported(test_dict["dataset_name"]) is test_dict["expected_success"]
