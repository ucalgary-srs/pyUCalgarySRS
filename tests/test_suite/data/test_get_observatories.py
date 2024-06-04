import pytest
import pyucalgarysrs


@pytest.mark.data_observatories
@pytest.mark.parametrize("instrument_array", ["themis_asi", "trex_rgb", "trex_nir", "trex_blue", "rego"])
def test_get_sites(srs, instrument_array):
    # get all sites
    sites = srs.data.list_observatories(instrument_array)
    assert len(sites) > 0

    # check serialization
    for d in sites:
        assert isinstance(d, pyucalgarysrs.Observatory) is True


@pytest.mark.data_observatories
@pytest.mark.parametrize("test_dict", [
    {
        "instrument_array": "themis_asi",
        "uid": "atha",
        "expected_results": 1
    },
    {
        "instrument_array": "themis_asi",
        "uid": "ATHA",
        "expected_results": 1
    },
    {
        "instrument_array": "rego",
        "uid": "fs",
        "expected_results": 2
    },
    {
        "instrument_array": "themis_asi",
        "uid": "bad",
        "expected_results": 0
    },
])
def test_get_datasets_filter_name(srs, test_dict):
    sites = srs.data.list_observatories(test_dict["instrument_array"], uid=test_dict["uid"])
    assert len(sites) == test_dict["expected_results"]
