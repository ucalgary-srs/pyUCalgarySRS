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
import pytest
import pyucalgarysrs
import warnings


@pytest.mark.data_download
@pytest.mark.parametrize("test_dict", [
    {
        "request": {
            "dataset_name": "THEMIS_ASI_RAW",
            "start": datetime.datetime(2023, 1, 3, 6, 0, 0),
            "end": datetime.datetime(2023, 1, 3, 6, 9, 0),
            "site_uid": "atha",
            "device_uid": None,
        },
        "expected_file_count": 10,
    },
    {
        "request": {
            "dataset_name": "TREX_RGB_DAILY_KEOGRAM",
            "start": datetime.datetime(2023, 1, 15, 0, 0, 0),
            "end": datetime.datetime(2023, 1, 19, 23, 59, 59),
            "site_uid": "gill",
            "device_uid": None,
        },
        "expected_file_count": 5,
    },
    {
        "request": {
            "dataset_name": "TREX_RGB_DAILY_KEOGRAM",
            "start": datetime.datetime(2023, 1, 15, 0, 0, 0),
            "end": datetime.datetime(2023, 1, 19, 23, 59, 59),
            "site_uid": "gill",
            "device_uid": "rgb-04"
        },
        "expected_file_count": 5,
    },
])
def test_download(srs, capsys, test_dict):
    # download data
    download_obj = srs.data.download(
        test_dict["request"]["dataset_name"],
        test_dict["request"]["start"],
        test_dict["request"]["end"],
        site_uid=test_dict["request"]["site_uid"],
        device_uid=test_dict["request"]["device_uid"],
        n_parallel=1,
        overwrite=True,
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == test_dict["expected_file_count"]

    # check that the files exist
    total_bytes_found = 0
    for f in download_obj.filenames:
        assert os.path.exists(f)
        total_bytes_found += os.path.getsize(f)
    assert total_bytes_found == download_obj.total_bytes

    # check __str__ and __repr__ for FileDownloadResult type
    print_str = str(download_obj)
    assert print_str != ""
    download_obj.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.data_download
def test_download_no_overwrite(srs, capsys):
    # download data
    #
    # NOTE: we do this twice so that the data is downloaded, and then
    # the overwrite logic is run through.
    download_obj = srs.data.download(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 3, 1, 6, 0, 0),
        datetime.datetime(2023, 3, 1, 8, 59, 59),
        site_uid="gill",
        n_parallel=1,
        overwrite=False,
    )
    download_obj = srs.data.download(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 3, 1, 6, 0, 0),
        datetime.datetime(2023, 3, 1, 8, 59, 59),
        site_uid="gill",
        n_parallel=1,
        overwrite=False,
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)

    # check __str__ and __repr__ for FileDownloadResult type
    print_str = str(download_obj)
    assert print_str != ""
    download_obj.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.data_download
def test_download_timeout(srs, capsys):
    # download data
    download_obj = srs.data.download(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 3, 1, 6, 0, 0),
        datetime.datetime(2023, 3, 1, 8, 59, 59),
        site_uid="gill",
        n_parallel=1,
        overwrite=True,
        timeout=5,
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True

    # check __str__ and __repr__ for FileDownloadResult type
    print_str = str(download_obj)
    assert print_str != ""
    download_obj.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)


@pytest.mark.data_download
def test_download_no_data(srs, capsys):
    # download data that doesn't exist, a warning should appear
    with warnings.catch_warnings(record=True) as w:
        # download data
        download_obj = srs.data.download(
            "TREX_RGB_HOURLY_KEOGRAM",
            datetime.datetime(2010, 1, 1, 6, 0, 0),
            datetime.datetime(2010, 1, 1, 8, 59, 59),
            site_uid="gill",
            n_parallel=1,
            timeout=5,
        )

    # check warning
    assert len(w) == 1
    assert issubclass(w[-1].category, UserWarning)
    assert "No data found to download" in str(w[-1].message)

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == 0

    # check __str__ and __repr__ for FileDownloadResult type
    print_str = str(download_obj)
    assert print_str != ""
    download_obj.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""
