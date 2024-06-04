import os
import datetime
import pytest
import pyucalgarysrs

ALL_TESTS = [
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
]


@pytest.mark.data_download
@pytest.mark.parametrize("test_dict", ALL_TESTS)
def test_download(srs, test_dict):
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


@pytest.mark.data_download
def test_download_no_overwrite(srs):
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


@pytest.mark.data_download
def test_download_timeout(srs):
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

    # check that the files exist
    for f in download_obj.filenames:
        assert os.path.exists(f)
