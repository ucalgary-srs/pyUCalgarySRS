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
from pyucalgarysrs import Skymap, SRSError, Data
from ...conftest import find_dataset

# globals
DATA_DIR = "%s/../../../test_data/read_skymap" % (os.path.dirname(os.path.realpath(__file__)))


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
    {
        "filename": "spect_skymap_luck_20230424-+_v01.sav",
        "dataset_name": "TREX_SPECT_SKYMAP_IDLSAV",
    },
])
@pytest.mark.data_read
def test_read_skymap_single_file(srs, capsys, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # read file
    data = srs.data.read(dataset, "%s/%s" % (DATA_DIR, test_dict["filename"]))

    # check return type
    assert isinstance(data, Data) is True
    assert isinstance(data.data, list) is True
    for item in data.data:
        assert isinstance(item, Skymap) is True

    # check __str__ and __repr__ for Data type
    print_str = str(data)
    assert print_str != ""
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check __str__ and __repr__ for Skymap type
    print_str = str(data.data[0])
    assert print_str != ""

    # check __str__ and __repr__ for SkymapGeneration type
    print_str = str(data.data[0].generation_info)
    assert print_str != ""

    # check pretty print methods
    data.data[0].pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""
    data.data[0].generation_info.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check precalculated altitudes method
    assert len(data.data[0].get_precalculated_altitudes()) > 0


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
def test_read_skymap_multiple_files(srs, all_datasets, test_dict, capsys):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    # read files
    data = srs.data.read(dataset, file_list)

    # check return type
    assert isinstance(data, Data) is True
    assert isinstance(data.data, list) is True
    for item in data.data:
        assert isinstance(item, Skymap) is True

    # check __str__ and __repr__ for Data type
    print_str = str(data)
    assert print_str != ""
    data.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


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
def test_read_skymap_n_parallel(srs, all_datasets, test_dict):
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
    assert isinstance(data, Data) is True
    assert isinstance(data.data, list) is True
    for item in data.data:
        assert isinstance(item, Skymap) is True


@pytest.mark.data_read
def test_read_skymap_badperms_file(srs):
    # set filename
    f = "%s/themis_skymap_gill_20210308-+_v02.sav" % (DATA_DIR)
    os.chmod(f, 0o000)

    # read file and check problematic files (not quiet mode)
    with pytest.raises(SRSError) as e_info:
        srs.data.readers.read_skymap(f, quiet=False)
    assert "Error reading skymap file" in str(e_info)

    # read file and check problematic files (quiet mode)
    with pytest.raises(SRSError) as e_info:
        srs.data.readers.read_skymap(f, quiet=True)
    assert "Error reading skymap file" in str(e_info)

    # change perms back
    os.chmod(f, 0o644)


@pytest.mark.parametrize("test_dict", [
    {
        "filenames": [
            "themis_skymap_atha_20070301-20090522_vXX.sav",
        ],
        "dataset_name": "THEMIS_ASI_SKYMAP_IDLSAV",
        "n_parallel": 1,
    },
])
@pytest.mark.data_read
def test_read_skymap_startend(srs, all_datasets, test_dict):
    # set dataset
    dataset = find_dataset(all_datasets, test_dict["dataset_name"])

    # build file list
    file_list = []
    for f in test_dict["filenames"]:
        file_list.append("%s/%s" % (DATA_DIR, f))

    with warnings.catch_warnings(record=True) as w:
        # read file
        data = srs.data.read(
            dataset,
            file_list,
            n_parallel=test_dict["n_parallel"],
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
        )

    # check return type
    assert isinstance(data, Data) is True
    assert isinstance(data.data, list) is True
    for item in data.data:
        assert isinstance(item, Skymap) is True

    # check that the warning appeared
    assert len(w) == 1
    assert issubclass(w[-1].category, UserWarning)
    assert "Reading of skymap files does not support the start_time or end_time parameters." in str(w[-1].message)
