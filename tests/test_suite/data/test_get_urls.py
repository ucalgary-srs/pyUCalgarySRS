import pytest
import datetime
import pyucalgarysrs

# NOTE: the following tests were taken verbatim from the SRS core API codebase
ALL_TESTS = [
    # ---------------
    # core tests
    # ---------------
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:59:59",
            "site_uid": "atha",
        },
        "expected_num_urls": 60,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-01T23:59:59",
            "site_uid": "atha",
        },
        "expected_num_urls": 814,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2023-01-01T05:50:00",
            "end": "2023-01-01T07:09:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 80,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:04:59",
        },
        "expected_num_urls": 5 * 11,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_DAILY_KEOGRAM_JPG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-01T23:59:59",
        },
        "expected_num_urls": 15,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:09:59",
            "site_uid": "gill",
            "include_total_bytes": True,
        },
        "expected_num_urls": 10,
        "expected_status": 200,
        "expected_total_bytes": 21668977,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_DAILY_KEOGRAM_JPG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-01T23:59:59",
            "include_total_bytes": True,
        },
        "expected_num_urls": 15,
        "expected_status": 200,
        "expected_total_bytes": 255415,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_THUMB32",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-02T05:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 25,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_SKYMAP_IDLSAV",
            "start": "2010-01-01T00:00:00",
            "end": "2010-12-31T23:59:59",
        },
        "expected_num_urls": 16,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_SKYMAP_IDLSAV",
            "start": "2010-01-01T00:00:00",
            "end": "2010-12-31T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 1,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_SKYMAP_IDLSAV",
            "start": "2010-01-01T00:00:00",
            "end": "2010-12-31T23:59:59",
            "site_uid": "gill",
            "device_uid": "themisXX",
        },
        "expected_num_urls": 1,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": True,
    },
    {
        "request": {
            "name": "THEMIS_ASI_SKYMAP_IDLSAV",
            "start": "2010-01-01T00:00:00",
            "end": "2010-12-31T23:59:59",
            "site_uid": "gill",
            "include_total_bytes": True,
        },
        "expected_num_urls": 1,
        "expected_status": 200,
        "expected_total_bytes": 3540232,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_SKYMAP_IDLSAV",
            "start": "2020-01-01T00:00:00",
            "end": "2020-01-01T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 0,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
            "start": "2014-01-01T00:00:00",
            "end": "2014-12-31T23:59:59",
        },
        "expected_num_urls": 6,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
            "start": "2014-01-01T00:00:00",
            "end": "2014-12-31T23:59:59",
        },
        "expected_num_urls": 6,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
            "start": "2014-01-01T00:00:00",
            "end": "2014-12-31T23:59:59",
            "include_total_bytes": True,
        },
        "expected_num_urls": 6,
        "expected_status": 200,
        "expected_total_bytes": 12591528,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
            "start": "2014-01-01T00:00:00",
            "end": "2014-12-31T23:59:59",
            "device_uid": "15649",
        },
        "expected_num_urls": 1,
        "expected_status": 200,
        "expected_total_bytes": 12591528,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
            "start": "2014-01-01T00:00:00",
            "end": "2014-12-31T23:59:59",
            "site_uid": "gill",  # testing that specifying site will not change results
        },
        "expected_num_urls": 6,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": True,
    },

    # --------------------------------------
    # core tests that will return no data
    # --------------------------------------
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2000-01-01T06:00:00",
            "end": "2000-01-01T06:09:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 0,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_DAILY_KEOGRAM_JPG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-01T23:59:59",
            "site_uid": "whit",
        },
        "expected_num_urls": 0,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },

    # ---------------------------------
    # core tests that will fail
    # ---------------------------------
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2023-01-01T06:09:59",
            "end": "2023-01-01T06:00:00",
            "site_uid": "gill",
        },
        "expected_num_urls": None,
        "expected_status": 400,
        "expected_error_message": "Start time is after the end time",
    },
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-02T00:00:00",
            "site_uid": "gill",
        },
        "expected_num_urls": None,
        "expected_status": 400,
        "expected_error_message": "Timeframe exceeds valid range; this dataset listing must be limited to one day",
    },
    {
        "request": {
            "name": "SOME_DATASET",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:09:00",
            "site_uid": "gill",
        },
        "expected_num_urls": None,
        "expected_status": 404,
        "expected_error_message": "Dataset name does not exist",
    },

    # ---------------------------------
    # tests for specific datasets
    # ---------------------------------
    {
        "request": {
            "name": "REGO_RAW",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 60,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_HOURLY_KEOGRAM_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_HOURLY_KEOGRAM_PNG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_HOURLY_KEOGRAM_JPG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_HOURLY_MONTAGE_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_HOURLY_MONTAGE_PNG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_HOURLY_MONTAGE_JPG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_DAILY_KEOGRAM_PGM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_DAILY_KEOGRAM_PNG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_DAILY_KEOGRAM_JPG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_DAILY_MONTAGE_PGM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_DAILY_MONTAGE_PNG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_DAILY_MONTAGE_JPG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_SKYMAP_IDLSAV",
            "start": "2020-01-01T00:00:00",
            "end": "2020-12-31T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 3,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
            "start": "2014-01-01T00:00:00",
            "end": "2020-12-31T23:59:59",
        },
        "expected_num_urls": 8,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "REGO_CALIBRATION_FLATFIELD_IDLSAV",
            "start": "2014-01-01T00:00:00",
            "end": "2020-12-31T23:59:59",
        },
        "expected_num_urls": 8,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_RAW",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 60,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_RAW_ROW2",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_RAW_WIDE",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-01T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 4,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_THUMB32",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_VEC1024",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_HOURLY_KEOGRAM_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_HOURLY_KEOGRAM_JPG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_HOURLY_MONTAGE_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_HOURLY_MONTAGE_JPG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_DAILY_KEOGRAM_PGM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 10,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_DAILY_KEOGRAM_JPG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_DAILY_MONTAGE_PGM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 10,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_DAILY_MONTAGE_JPG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_HOURLY_AVERAGE_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_HOURLY_AVERAGE_JPG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "THEMIS_ASI_SKYMAP_IDLSAV",
            "start": "2020-01-01T00:00:00",
            "end": "2020-12-31T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_RGB_RAW_BURST",
            "start": "2023-01-01T10:00:00",
            "end": "2023-01-01T10:59:59",
            "site_uid": "atha",
        },
        "expected_num_urls": 45,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_RGB_RAW_NOMINAL",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 60,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_RGB_HOURLY_KEOGRAM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_RGB_HOURLY_MONTAGE",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_RGB_DAILY_KEOGRAM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_RGB_DAILY_MONTAGE",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_RAW",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 60,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_HOURLY_KEOGRAM_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_HOURLY_KEOGRAM_PNG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_HOURLY_MONTAGE_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_HOURLY_MONTAGE_PNG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_DAILY_KEOGRAM_PGM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_DAILY_KEOGRAM_PNG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_DAILY_MONTAGE_PGM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_DAILY_MONTAGE_PNG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_SKYMAP_IDLSAV",
            "start": "2020-01-01T00:00:00",
            "end": "2020-12-31T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 3,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_CALIBRATION_RAYLEIGHS_IDLSAV",
            "start": "2020-01-01T00:00:00",
            "end": "2023-12-31T23:59:59",
        },
        "expected_num_urls": 7,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_NIR_CALIBRATION_FLATFIELD_IDLSAV",
            "start": "2020-01-01T00:00:00",
            "end": "2023-12-31T23:59:59",
        },
        "expected_num_urls": 7,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_RAW",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T06:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 60,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_HOURLY_KEOGRAM_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_HOURLY_KEOGRAM_PNG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_HOURLY_MONTAGE_PGM",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_HOURLY_MONTAGE_PNG",
            "start": "2023-01-01T06:00:00",
            "end": "2023-01-01T07:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 2,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_DAILY_KEOGRAM_PGM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_DAILY_KEOGRAM_PNG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_DAILY_MONTAGE_PGM",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
    {
        "request": {
            "name": "TREX_BLUE_DAILY_MONTAGE_PNG",
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-05T23:59:59",
            "site_uid": "gill",
        },
        "expected_num_urls": 5,
        "expected_status": 200,
        "expected_error_message": None,
        "expected_warning": False,
    },
]


@pytest.mark.data_geturls
@pytest.mark.parametrize("test_dict", ALL_TESTS)
def test_get_urls(srs, test_dict):
    # get urls
    url_obj = None
    if (test_dict["expected_status"] == 200):
        if (test_dict["expected_warning"] is True):
            with pytest.warns(UserWarning) as w_info:
                url_obj = srs.data.get_urls(
                    test_dict["request"]["name"],
                    datetime.datetime.fromisoformat(test_dict["request"]["start"]),
                    datetime.datetime.fromisoformat(test_dict["request"]["end"]),
                    site_uid=None if "site_uid" not in test_dict["request"] else test_dict["request"]["site_uid"],
                    device_uid=None if "device_uid" not in test_dict["request"] else test_dict["request"]["device_uid"],
                )
            assert len(w_info) == 1
            assert "field is not used when filtering in this dataset" in str(w_info[0].message).lower()
        else:
            url_obj = srs.data.get_urls(
                test_dict["request"]["name"],
                datetime.datetime.fromisoformat(test_dict["request"]["start"]),
                datetime.datetime.fromisoformat(test_dict["request"]["end"]),
                site_uid=None if "site_uid" not in test_dict["request"] else test_dict["request"]["site_uid"],
                device_uid=None if "device_uid" not in test_dict["request"] else test_dict["request"]["device_uid"],
            )
    else:
        with pytest.raises(pyucalgarysrs.SRSAPIError) as e_info:
            url_obj = srs.data.get_urls(
                test_dict["request"]["name"],
                datetime.datetime.fromisoformat(test_dict["request"]["start"]),
                datetime.datetime.fromisoformat(test_dict["request"]["end"]),
                site_uid=test_dict["request"]["site_uid"],
                device_uid=None if "device_uid" not in test_dict["request"] else test_dict["request"]["device_uid"],
            )
            try:
                if (test_dict["expected_error_message"][0] == '~'):
                    # partial match
                    assert test_dict["expected_error_message"][1:] in str(e_info)
                elif test_dict["expected_error_message"][0] == '=':
                    # exact match
                    assert test_dict["expected_error_message"][1:] == str(e_info)
                else:
                    # default to exact match, but assuming there is no special character at the start
                    assert test_dict["expected_error_message"] == str(e_info)
            except Exception as e:
                raise Exception("Test unexpectedly failed: %s" % (str(e_info))) from e
        return
    assert url_obj is not None

    # check response type
    assert isinstance(url_obj, pyucalgarysrs.FileListingResponse) is True

    # check dataset matches
    assert url_obj.dataset.name == test_dict["request"]["name"]

    # check files were found
    assert len(url_obj.urls) == test_dict["expected_num_urls"]
