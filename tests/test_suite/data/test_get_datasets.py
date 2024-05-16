import pytest
import pyucalgarysrs

ALL_FILTER_TESTS = [
    {
        "name": "THEMIS",
        "expected_zero_results": False,
    },
    {
        "name": "REGO",
        "expected_zero_results": False,
    },
    {
        "name": "DAILY_KEOGRAM",
        "expected_zero_results": False,
    },
    {
        "name": "HOURLY_KEOGRAM",
        "expected_zero_results": False,
    },
    {
        "name": "KEOGRAM",
        "expected_zero_results": False,
    },
    {
        "name": "CALIBRATION",
        "expected_zero_results": False,
    },
    {
        "name": "SKYMAP",
        "expected_zero_results": False,
    },
    {
        "name": "",
        "expected_zero_results": False,
    },
    {
        "name": "some_bad_name",
        "expected_zero_results": True,
    },
]


@pytest.mark.data_datasets
def test_get_datasets(srs, capsys):
    # get all datasets
    datasets = srs.data.list_datasets()

    # check serialization
    for d in datasets:
        assert isinstance(d, pyucalgarysrs.Dataset) is True

    # check acknowledgement print method
    datasets[0].show_acknowledgement_info()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != "" and "DOI" in captured_stdout


@pytest.mark.data_datasets
@pytest.mark.parametrize("test_dict", ALL_FILTER_TESTS)
def test_get_datasets_filter_name(srs, test_dict):
    # get datasets
    datasets = srs.data.list_datasets(name=test_dict["name"])

    # check if we expect any datasets to be found
    if (test_dict["expected_zero_results"] is True):
        assert len(datasets) == 0
        return
    else:
        assert len(datasets) > 0

    # check filter
    for d in datasets:
        assert test_dict["name"] in d.name
