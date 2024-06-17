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

ALL_TESTS = [
    {
        "request": {
            "dataset_name": "THEMIS_ASI_RAW",
            "start": datetime.datetime(2023, 1, 2, 6, 0, 0),
            "end": datetime.datetime(2023, 1, 2, 6, 9, 0),
            "site_uid": "atha",
        },
        "expected_success": True,
        "expected_url_count": 10,
        "expected_error_message": None,
    },
    {
        "request": {
            "dataset_name": "TREX_RGB_DAILY_KEOGRAM",
            "start": datetime.datetime(2023, 1, 10, 0, 0, 0),
            "end": datetime.datetime(2023, 1, 14, 23, 59, 59),
            "site_uid": "gill",
        },
        "expected_success": True,
        "expected_url_count": 5,
        "expected_error_message": None,
    },
]


@pytest.mark.data_download
@pytest.mark.parametrize("test_dict", ALL_TESTS)
def test_download_using_urls(srs, test_dict):
    # get urls
    file_listing_obj = srs.data.get_urls(
        test_dict["request"]["dataset_name"],
        test_dict["request"]["start"],
        test_dict["request"]["end"],
        site_uid=test_dict["request"]["site_uid"],
    )
    assert file_listing_obj.count == test_dict["expected_url_count"]

    # download urls
    download_obj = srs.data.download_using_urls(
        file_listing_obj,
        n_parallel=1,
        overwrite=True,
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == file_listing_obj.count

    # check that the files exist
    total_bytes_found = 0
    for f in download_obj.filenames:
        assert os.path.exists(f)
        total_bytes_found += os.path.getsize(f)
    assert total_bytes_found == download_obj.total_bytes


@pytest.mark.data_download
def test_download_using_urls_no_overwrite(srs):
    # get urls
    file_listing_obj = srs.data.get_urls(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 2, 1, 6, 0, 0),
        datetime.datetime(2023, 2, 1, 8, 59, 59),
        site_uid="gill",
    )

    # download urls
    download_obj = srs.data.download_using_urls(
        file_listing_obj,
        n_parallel=1,
        overwrite=False,
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == file_listing_obj.count

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)


@pytest.mark.data_download
def test_download_using_urls_n_parallel(srs):
    # get urls
    file_listing_obj = srs.data.get_urls(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 2, 2, 6, 0, 0),
        datetime.datetime(2023, 2, 2, 8, 59, 59),
        site_uid="gill",
    )

    # download urls
    download_obj = srs.data.download_using_urls(
        file_listing_obj,
        n_parallel=2,
        overwrite=True,
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == file_listing_obj.count

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)


@pytest.mark.data_download
def test_download_using_urls_progress_bar_disable(srs):
    # get urls
    file_listing_obj = srs.data.get_urls(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 2, 3, 6, 0, 0),
        datetime.datetime(2023, 2, 3, 8, 59, 59),
        site_uid="gill",
    )

    # download urls
    download_obj = srs.data.download_using_urls(
        file_listing_obj,
        n_parallel=1,
        overwrite=True,
        progress_bar_disable=True,
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == file_listing_obj.count

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)


@pytest.mark.data_download
def test_download_using_urls_progress_bar_ncols(srs):
    # get urls
    file_listing_obj = srs.data.get_urls(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 2, 4, 6, 0, 0),
        datetime.datetime(2023, 2, 4, 8, 59, 59),
        site_uid="gill",
    )

    # download urls
    download_obj = srs.data.download_using_urls(
        file_listing_obj,
        n_parallel=1,
        overwrite=True,
        progress_bar_ncols=100,
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == file_listing_obj.count

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)


@pytest.mark.data_download
def test_download_using_urls_progress_bar_ascii(srs):
    # get urls
    file_listing_obj = srs.data.get_urls(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 2, 5, 6, 0, 0),
        datetime.datetime(2023, 2, 5, 8, 59, 59),
        site_uid="gill",
    )

    # download urls
    download_obj = srs.data.download_using_urls(
        file_listing_obj,
        n_parallel=1,
        overwrite=True,
        progress_bar_ascii=" +",
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == file_listing_obj.count

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)


@pytest.mark.data_download
def test_download_using_urls_progress_bar_desc(srs):
    # get urls
    file_listing_obj = srs.data.get_urls(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 2, 5, 6, 0, 0),
        datetime.datetime(2023, 2, 5, 8, 59, 59),
        site_uid="gill",
    )

    # download urls
    download_obj = srs.data.download_using_urls(
        file_listing_obj,
        n_parallel=1,
        overwrite=True,
        progress_bar_desc="Some download description",
    )

    # check download object
    assert isinstance(download_obj, pyucalgarysrs.FileDownloadResult) is True
    assert download_obj.count == file_listing_obj.count

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)


@pytest.mark.data_download
def test_download_using_urls_bad_url(srs):
    # get urls
    file_listing_obj = srs.data.get_urls(
        "TREX_RGB_HOURLY_KEOGRAM",
        datetime.datetime(2023, 2, 5, 6, 0, 0),
        datetime.datetime(2023, 2, 5, 8, 59, 59),
        site_uid="gill",
    )

    # hack one of the urls
    file_listing_obj.urls[0] = "https://data.phys.ucalgary.ca/pyucalgarysrs-testing/fake_url.txt"

    # get urls, expecting a failure
    with pytest.raises(pyucalgarysrs.SRSDownloadError) as e_info:
        srs.data.download_using_urls(
            file_listing_obj,
            n_parallel=1,
            overwrite=True,
            progress_bar_desc="Some download description",
        )
        assert "HTTP error 404 when downloading" in str(e_info)
