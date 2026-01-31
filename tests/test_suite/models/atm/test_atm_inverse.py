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

import pytest
import pyucalgarysrs
import datetime
import warnings

# NOTE: the following tests were taken verbatim from the SRS core API codebase
ALL_TESTS = [
    # ---------------
    # core tests
    # ---------------
    {
        "request": {
            "timestamp": "2023-03-24T07:00:00",
            "geodetic_latitude": 58.0,
            "geodetic_longitude": -105.0,
            "nrlmsis_model_version": "2.0",
            "intensity_4278": 2302.6,
            "intensity_5577": 11339.5,
            "intensity_6300": 528.3,
            "intensity_8446": 427.4,
            "precipitation_flux_spectral_type": "gaussian",
            "output": {
                "energy_flux": True,
                "mean_energy": True,
                "oxygen_correction_factor": True,
            },
            "no_cache": True,
        },
        "expected_status": 200,
        "expected_error_message": None,
    },
    {
        "request": {
            "timestamp": "2023-03-24T07:00:00",
            "geodetic_latitude": 58.0,
            "geodetic_longitude": -105.0,
            "nrlmsis_model_version": "2.0",
            "intensity_4278": 2302.6,
            "intensity_5577": 11339.5,
            "intensity_6300": 528.3,
            "intensity_8446": 427.4,
            "precipitation_flux_spectral_type": "gaussian",
            "output": {
                "energy_flux": True,
                "mean_energy": True,
                "oxygen_correction_factor": True,
            },
            "no_cache": False,
        },
        "expected_status": 200,
        "expected_error_message": None,
    },
    {
        "request": {
            "timestamp": "2023-03-24T07:00:00",
            "geodetic_latitude": 58.0,
            "geodetic_longitude": -105.0,
            "nrlmsis_model_version": "2.0",
            "intensity_4278": 2302.6,
            "intensity_5577": 11339.5,
            "intensity_6300": 528.3,
            "intensity_8446": 427.4,
            "precipitation_flux_spectral_type": "gaussian",
            "output": {
                "energy_flux": True,
                "mean_energy": True,
                "oxygen_correction_factor": True,
            },
        },
        "expected_status": 200,
        "expected_error_message": None,
    },
    {
        "request": {
            "timestamp": "2023-03-24T07:00:00",
            "geodetic_latitude": 58.0,
            "geodetic_longitude": -105.0,
            "nrlmsis_model_version": "2.0",
            "intensity_4278": 2302.6,
            "intensity_5577": 11339.5,
            "intensity_6300": 528.3,
            "intensity_8446": 427.4,
            "precipitation_flux_spectral_type": "gaussian",
            "output": {
                "energy_flux": True,
                "mean_energy": False,
                "oxygen_correction_factor": False,
            },
        },
        "expected_status": 200,
        "expected_error_message": None,
    },
    {
        "request": {
            "timestamp": "2023-03-24T07:00:00",
            "geodetic_latitude": 58.0,
            "geodetic_longitude": -105.0,
            "nrlmsis_model_version": "2.0",
            "intensity_4278": 2302.6,
            "intensity_5577": 11339.5,
            "intensity_6300": 528.3,
            "intensity_8446": 427.4,
            "precipitation_flux_spectral_type": "maxwellian",
            "output": {
                "energy_flux": True,
                "mean_energy": True,
                "oxygen_correction_factor": True,
            },
            "no_cache": True,
        },
        "expected_status": 200,
        "expected_error_message": None,
    },
    {
        "request": {
            "timestamp": "2023-03-24T07:00:00",
            "geodetic_latitude": 58.0,
            "geodetic_longitude": -105.0,
            "nrlmsis_model_version": "2.0",
            "intensity_4278": 2302.6,
            "intensity_5577": 11339.5,
            "intensity_6300": 528.3,
            "intensity_8446": 427.4,
            "precipitation_flux_spectral_type": "gaussian",
            "no_cache": True,
        },
        "expected_status": 200,
        "expected_error_message": None,
    },

    # -------------------------
    # tests expected to fail
    # -------------------------
    {
        "request": {
            "timestamp": "2000-01-01T06:00:00",
            "geodetic_latitude": 58.0,
            "geodetic_longitude": -105.0,
            "nrlmsis_model_version": "2.0",
            "intensity_4278": 2302.6,
            "intensity_5577": 11339.5,
            "intensity_6300": 528.3,
            "intensity_8446": 427.4,
            "no_cache": True,
        },
        "expected_status": 400,
        "expected_error_message": "Supplied date is not supported at this time",
    },
]


def __do_function(srs_obj, test_dict):
    # set up output object
    output_obj = pyucalgarysrs.ATMInverseOutputFlags()
    if ("output" in test_dict["request"]):
        for key, value in test_dict["request"]["output"].items():
            setattr(output_obj, key, value)

    # set up precipitation_flux_spectral_type param
    precipitation_flux_spectral_type = "gaussian"
    if ("precipitation_flux_spectral_type" in test_dict["request"]):
        precipitation_flux_spectral_type = test_dict["request"]["precipitation_flux_spectral_type"]

    # set up no_cache param
    no_cache = False
    if ("no_cache" in test_dict["request"]):
        no_cache = test_dict["request"]["no_cache"]

    # set up atm model version
    atm_model_version = "2.0"
    if ("atm_model_version" in test_dict["request"]):
        atm_model_version = test_dict["request"]["atm_model_version"]

    # do calculation
    result = srs_obj.models.atm.inverse(
        datetime.datetime.fromisoformat(test_dict["request"]["timestamp"]),
        test_dict["request"]["geodetic_latitude"],
        test_dict["request"]["geodetic_longitude"],
        test_dict["request"]["intensity_4278"],
        test_dict["request"]["intensity_5577"],
        test_dict["request"]["intensity_6300"],
        test_dict["request"]["intensity_8446"],
        output_obj,
        precipitation_flux_spectral_type=precipitation_flux_spectral_type,
        nrlmsis_model_version=test_dict["request"]["nrlmsis_model_version"],
        atm_model_version=atm_model_version,
        no_cache=no_cache,
    )

    # return
    return result, output_obj


@pytest.mark.parametrize("test_dict", ALL_TESTS)
@pytest.mark.atm
def test_atm_inverse(srs, test_dict):
    # change test if it's expected to fail
    if (test_dict["expected_status"] != 200):
        with pytest.raises(pyucalgarysrs.SRSAPIError) as e_info:
            __do_function(srs, test_dict)
        assert test_dict["expected_error_message"] in (str(e_info))
        return

    # make request
    result, output_obj = __do_function(srs, test_dict)

    # check return parameters
    excluded_items = ["set_all_true", "set_all_false"]
    for output_var in dir(output_obj):
        if (output_var.startswith("__") or output_var in excluded_items):
            continue
        if (getattr(output_obj, output_var) is True):
            # should have gotten data for this var, check the result
            if ("atm_model_version" in test_dict and test_dict["atm_model_version"] == "1.0"):
                assert getattr(result, output_var) is not None
            else:
                if (output_var in ["energy_flux", "oxygen_correction_factor", "mean_energy"]):
                    assert getattr(result, output_var) is not None
                else:
                    assert getattr(result, output_var) is None
        else:
            # should not have gotten data for this var, check the result
            assert getattr(result, output_var) is None


@pytest.mark.atm
def test_atm_inverse_schema_atm_inverse_output_flags():
    # init
    excluded_items = ["set_all_true", "set_all_false"]

    # create object
    output_obj = pyucalgarysrs.ATMInverseOutputFlags()

    # check set_all_true method
    output_obj.set_all_true()
    for var_name in dir(output_obj):
        if (var_name.startswith("__") or var_name in excluded_items):
            continue
        assert getattr(output_obj, var_name) is True

    # check set_all_false method
    output_obj.set_all_false()
    for var_name in dir(output_obj):
        if (var_name.startswith("__") or var_name in excluded_items):
            continue
        assert getattr(output_obj, var_name) is False


@pytest.mark.atm
def test_atm_inverse_schema_atm_inverse_result(srs, capsys):
    # init
    test_dict = ALL_TESTS[0]
    for key in test_dict["request"]["output"].keys():
        test_dict["request"]["output"][key] = True

    # do a request
    result, _ = __do_function(srs, test_dict)

    # check pretty_print method
    result.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check __str__ and __repr__
    print_str = str(result)
    assert print_str != ""


@pytest.mark.atm
def test_special_logic_keyword(srs, capsys):
    # set request
    request_dict = {
        "request": {
            "timestamp": "2025-01-01T12:00:00",
            "geodetic_latitude": 58.0,
            "geodetic_longitude": -105.0,
            "nrlmsis_model_version": "2.0",
            "intensity_4278": 2302.6,
            "intensity_5577": 11339.5,
            "intensity_6300": 528.3,
            "intensity_8446": 427.4,
            "precipitation_flux_spectral_type": "gaussian",
            "special_logic_keyword": "shill_20250910",
            "output": {
                "energy_flux": True,
                "mean_energy": True,
                "oxygen_correction_factor": True,
            },
            "no_cache": True,
        },
        "expected_status": 200,
        "expected_error_message": None,
    }

    # do a request
    result, _ = __do_function(srs, request_dict)

    # check pretty_print method
    result.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.atm
def test_atm_inverse_model_version_warning(srs, capsys):
    # set request
    request_dict = ALL_TESTS[0]
    request_dict["request"]["atm_model_version"] = "1.0"

    # do a request
    with warnings.catch_warnings(record=True) as w:
        result, _ = __do_function(srs, request_dict)

    # check warning
    assert len(w) == 1
    assert issubclass(w[-1].category, UserWarning)
    assert "Using ATM version 1.0 is no longer supported in this library" in str(w[-1].message)

    # check pretty_print method
    result.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""
