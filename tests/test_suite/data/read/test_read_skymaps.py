import os
import pytest
from pyucalgarysrs.data import Skymap
from ...conftest import find_dataset

# globals
DATA_DIR = "%s/../../../test_data/read_skymaps" % (os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize("test_dict", [
    {
        "filename": "themis_skymap_atha_20070301-20090522_vXX.sav",
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
    },
    {
        "filename": "themis_skymap_atha_20230115-+_v02.sav",
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
    },
    {
        "filename": "rego_skymap_atha_20140718-+_v01.sav",
        "dataset_name": "REGO_SKYMAP_IDLSAV",
    },
    {
        "filename": "rego_skymap_luck_20230707-+_v01.sav",
        "dataset_name": "REGO_SKYMAP_IDLSAV",
    },
    {
        "filename": "nir_skymap_atha_20220920-+_v01.sav",
        "dataset_name": "TREX_NIR_SKYMAP_IDLSAV",
    },
    {
        "filename": "rgb_skymap_atha_20231003-+_v01.sav",
        "dataset_name": "TREX_RGB_SKYMAP_IDLSAV",
    },
])
@pytest.mark.data_read
def test_read_skymaps_single_file(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # read file
    data = srs.data.read(dataset, "%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check return type
    assert isinstance(data, list) is True
    assert isinstance(data[0], Skymap) is True


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "themis_skymap_atha_20070301-20090522_vXX.sav",
        ],
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
    },
    {
        "filenames": [
            "themis_skymap_atha_20070301-20090522_vXX.sav",
            "themis_skymap_atha_20230115-+_v02.sav",
        ],
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
    },
])
@pytest.mark.data_read
def test_read_skymaps_multiple_files(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read files
    data = srs.data.read(dataset, file_list)

    # check return type
    assert isinstance(data, list) is True
    for item in data:
        assert isinstance(item, Skymap) is True


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "themis_skymap_atha_20070301-20090522_vXX.sav",
        ],
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
        "n_parallel": 1,
    },
    {
        "filenames": [
            "themis_skymap_atha_20070301-20090522_vXX.sav",
        ],
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
        "n_parallel": 2,
    },
    {
        "filenames": [
            "themis_skymap_atha_20070301-20090522_vXX.sav",
            "themis_skymap_atha_20230115-+_v02.sav",
        ],
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
        "n_parallel": 2,
    },
])
@pytest.mark.data_read
def test_read_skymaps_single_file_n_parallel(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

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

    # check return type
    assert isinstance(data, list) is True
    for item in data:
        assert isinstance(item, Skymap) is True
