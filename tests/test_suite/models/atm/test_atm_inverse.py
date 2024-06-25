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
            "atmospheric_attenuation_correction": False,
            "output": {
                "altitudes": False,
                "energy_flux": True,
                "characteristic_energy": True,
                "oxygen_correction_factor": True,
                "height_integrated_rayleighs_4278": False,
                "height_integrated_rayleighs_5577": False,
                "height_integrated_rayleighs_6300": False,
                "height_integrated_rayleighs_8446": False,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_oplus_density": False,
                "plasma_noplus_density": False,
                "plasma_ionisation_rate": False,
                "plasma_electron_temperature": False,
                "plasma_ion_temperature": False,
                "plasma_pederson_conductivity": False,
                "plasma_hall_conductivity": False,
                "neutral_o2_density": False,
                "neutral_o_density": False,
                "neutral_n2_density": False,
                "neutral_n_density": False,
                "neutral_temperature": False
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
            "atmospheric_attenuation_correction": True,
            "output": {
                "altitudes": False,
                "energy_flux": True,
                "characteristic_energy": True,
                "oxygen_correction_factor": True,
                "height_integrated_rayleighs_4278": False,
                "height_integrated_rayleighs_5577": False,
                "height_integrated_rayleighs_6300": False,
                "height_integrated_rayleighs_8446": False,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_oplus_density": False,
                "plasma_noplus_density": False,
                "plasma_ionisation_rate": False,
                "plasma_electron_temperature": False,
                "plasma_ion_temperature": False,
                "plasma_pederson_conductivity": False,
                "plasma_hall_conductivity": False,
                "neutral_o2_density": False,
                "neutral_o_density": False,
                "neutral_n2_density": False,
                "neutral_n_density": False,
                "neutral_temperature": False
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
            "atmospheric_attenuation_correction": False,
            "output": {
                "altitudes": False,
                "energy_flux": True,
                "characteristic_energy": True,
                "oxygen_correction_factor": True,
                "height_integrated_rayleighs_4278": False,
                "height_integrated_rayleighs_5577": False,
                "height_integrated_rayleighs_6300": False,
                "height_integrated_rayleighs_8446": False,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_oplus_density": False,
                "plasma_noplus_density": False,
                "plasma_ionisation_rate": False,
                "plasma_electron_temperature": False,
                "plasma_ion_temperature": False,
                "plasma_pederson_conductivity": False,
                "plasma_hall_conductivity": False,
                "neutral_o2_density": False,
                "neutral_o_density": False,
                "neutral_n2_density": False,
                "neutral_n_density": False,
                "neutral_temperature": False
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
            "atmospheric_attenuation_correction": False,
            "output": {
                "altitudes": False,
                "energy_flux": True,
                "characteristic_energy": True,
                "oxygen_correction_factor": True,
                "height_integrated_rayleighs_4278": False,
                "height_integrated_rayleighs_5577": False,
                "height_integrated_rayleighs_6300": False,
                "height_integrated_rayleighs_8446": False,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_oplus_density": False,
                "plasma_noplus_density": False,
                "plasma_ionisation_rate": False,
                "plasma_electron_temperature": False,
                "plasma_ion_temperature": False,
                "plasma_pederson_conductivity": False,
                "plasma_hall_conductivity": False,
                "neutral_o2_density": False,
                "neutral_o_density": False,
                "neutral_n2_density": False,
                "neutral_n_density": False,
                "neutral_temperature": False
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
            "atmospheric_attenuation_correction": False,
            "output": {
                "altitudes": True,
                "energy_flux": True,
                "characteristic_energy": True,
                "oxygen_correction_factor": True,
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "emission_4278": True,
                "emission_5577": True,
                "emission_6300": True,
                "emission_8446": True,
                "plasma_electron_density": True,
                "plasma_o2plus_density": True,
                "plasma_oplus_density": True,
                "plasma_noplus_density": True,
                "plasma_ionisation_rate": True,
                "plasma_electron_temperature": True,
                "plasma_ion_temperature": True,
                "plasma_pederson_conductivity": True,
                "plasma_hall_conductivity": True,
                "neutral_o2_density": True,
                "neutral_o_density": True,
                "neutral_n2_density": True,
                "neutral_n_density": True,
                "neutral_temperature": True
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
            "precipitation_flux_spectral_type": "maxwellian",
            "atmospheric_attenuation_correction": False,
            "output": {
                "altitudes": True,
                "energy_flux": True,
                "characteristic_energy": True,
                "oxygen_correction_factor": True,
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "emission_4278": True,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_oplus_density": False,
                "plasma_noplus_density": False,
                "plasma_ionisation_rate": False,
                "plasma_electron_temperature": False,
                "plasma_ion_temperature": False,
                "plasma_pederson_conductivity": False,
                "plasma_hall_conductivity": False,
                "neutral_o2_density": False,
                "neutral_o_density": False,
                "neutral_n2_density": False,
                "neutral_n_density": False,
                "neutral_temperature": False
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
            "atmospheric_attenuation_correction": False,
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
            "atmospheric_attenuation_correction": False,
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
        atmospheric_attenuation_correction=test_dict["request"]["atmospheric_attenuation_correction"],
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
            assert getattr(result, output_var) is not None
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
