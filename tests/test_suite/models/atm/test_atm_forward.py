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
import numpy as np
import datetime

# NOTE: the following tests were taken verbatim from the SRS core API codebase
ALL_TESTS = [
    # ---------------
    # core tests
    # ---------------
    {
        "request": {
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "output": {
                "altitudes": False,
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "height_integrated_rayleighs_lbh": True,
                "height_integrated_rayleighs_1304": True,
                "height_integrated_rayleighs_1356": True,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "emission_lbh": False,
                "emission_1304": False,
                "emission_1356": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_noplus_density": False,
                "plasma_oplus_density": False,
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
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "output": {
                "altitudes": False,
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "height_integrated_rayleighs_lbh": True,
                "height_integrated_rayleighs_1304": True,
                "height_integrated_rayleighs_1356": True,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "emission_lbh": False,
                "emission_1304": False,
                "emission_1356": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_noplus_density": False,
                "plasma_oplus_density": False,
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
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "output": {
                "altitudes": False,
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "height_integrated_rayleighs_lbh": True,
                "height_integrated_rayleighs_1304": True,
                "height_integrated_rayleighs_1356": True,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "emission_lbh": False,
                "emission_1304": False,
                "emission_1356": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_noplus_density": False,
                "plasma_oplus_density": False,
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
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "output": {
                "altitudes": False,
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "height_integrated_rayleighs_lbh": True,
                "height_integrated_rayleighs_1304": True,
                "height_integrated_rayleighs_1356": True,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "emission_lbh": False,
                "emission_1304": False,
                "emission_1356": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_noplus_density": False,
                "plasma_oplus_density": False,
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
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "output": {
                "altitudes": True,
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "height_integrated_rayleighs_lbh": True,
                "height_integrated_rayleighs_1304": True,
                "height_integrated_rayleighs_1356": True,
                "emission_4278": True,
                "emission_5577": True,
                "emission_6300": True,
                "emission_8446": True,
                "emission_lbh": True,
                "emission_1304": True,
                "emission_1356": True,
                "plasma_electron_density": True,
                "plasma_o2plus_density": True,
                "plasma_noplus_density": True,
                "plasma_oplus_density": True,
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
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "output": {
                "altitudes": False,
                "height_integrated_rayleighs_4278": False,
                "height_integrated_rayleighs_5577": False,
                "height_integrated_rayleighs_6300": False,
                "height_integrated_rayleighs_8446": False,
                "height_integrated_rayleighs_lbh": False,
                "height_integrated_rayleighs_1304": False,
                "height_integrated_rayleighs_1356": False,
                "emission_4278": False,
                "emission_5577": False,
                "emission_6300": False,
                "emission_8446": False,
                "emission_lbh": False,
                "emission_1304": False,
                "emission_1356": False,
                "plasma_electron_density": False,
                "plasma_o2plus_density": False,
                "plasma_noplus_density": False,
                "plasma_oplus_density": False,
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
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "no_cache": True,
        },
        "expected_status": 200,
        "expected_error_message": None,
    },
    {
        "request": {
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "00",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
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
            "timestamp": "2050-11-04T06:00:00",
            "geodetic_latitude": 100.0,
            "geodetic_longitude": 100.0,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "no_cache": True,
        },
        "expected_status": 400,
        "expected_error_message": "Timestamp only valid for up to the end of the previous day",
    },
    {
        "request": {
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 100.0,
            "geodetic_longitude": 100.0,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "no_cache": True,
        },
        "expected_status": 400,
        "expected_error_message": "Geodetic latitude must be between -90 and 90",
    },
    {
        "request": {
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 90.0,
            "geodetic_longitude": -200.0,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "no_cache": True,
        },
        "expected_status": 400,
        "expected_error_message": "Geodetic longitude must be between -180 and 180",
    },
]


def __do_function(srs_obj, test_dict):
    # set up output object
    output_obj = pyucalgarysrs.ATMForwardOutputFlags()
    if ("output" in test_dict["request"]):
        for key, value in test_dict["request"]["output"].items():
            setattr(output_obj, key, value)

    # set up custom spectrum
    custom_spectrum = None
    if ("custom_spectrum" in test_dict["request"]):
        custom_spectrum = test_dict["request"]["custom_spectrum"]

    # set up custom neutral profile
    custom_neutral_profile = None
    if ("custom_neutral_profile" in test_dict["request"]):
        custom_neutral_profile = test_dict["request"]["custom_neutral_profile"]

    # set up no_cache param
    no_cache = False
    if ("no_cache" in test_dict["request"]):
        no_cache = test_dict["request"]["no_cache"]

    # do calculation
    result = srs_obj.models.atm.forward(
        datetime.datetime.fromisoformat(test_dict["request"]["timestamp"]),
        test_dict["request"]["geodetic_latitude"],
        test_dict["request"]["geodetic_longitude"],
        output_obj,
        maxwellian_energy_flux=test_dict["request"]["maxwellian_energy_flux"],
        gaussian_energy_flux=test_dict["request"]["gaussian_energy_flux"],
        maxwellian_characteristic_energy=test_dict["request"]["maxwellian_characteristic_energy"],
        gaussian_peak_energy=test_dict["request"]["gaussian_peak_energy"],
        gaussian_spectral_width=test_dict["request"]["gaussian_spectral_width"],
        nrlmsis_model_version=test_dict["request"]["nrlmsis_model_version"],
        oxygen_correction_factor=test_dict["request"]["oxygen_correction_factor"],
        timescale_auroral=test_dict["request"]["timescale_auroral"],
        timescale_transport=test_dict["request"]["timescale_transport"],
        custom_spectrum=custom_spectrum,
        custom_neutral_profile=custom_neutral_profile,
        no_cache=no_cache,
    )

    # return
    return result, output_obj


@pytest.mark.parametrize("test_dict", ALL_TESTS)
@pytest.mark.atm
def test_atm_forward(srs, test_dict):
    # change test if it's expected to fail
    if (test_dict["expected_status"] != 200):
        with pytest.raises(pyucalgarysrs.SRSAPIError) as e_info:
            __do_function(srs, test_dict)
        assert test_dict["expected_error_message"] in (str(e_info))
        return

    # make request
    result, output_obj = __do_function(srs, test_dict)

    # check return parameters
    excluded_items = ["set_all_true", "set_all_false", "enable_only_height_integrated_rayleighs"]
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
def test_atm_forward_schema_atm_forward_output_flags():
    # init
    excluded_items = ["set_all_true", "set_all_false", "enable_only_height_integrated_rayleighs"]

    # create object
    output_obj = pyucalgarysrs.ATMForwardOutputFlags()

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

    # check enable_only_height_integrated_rayleighs method
    output_obj.enable_only_height_integrated_rayleighs()
    for var_name in dir(output_obj):
        if (var_name.startswith("__") or var_name in excluded_items):
            continue
        if ("height_integrated_rayleighs" in var_name):
            assert getattr(output_obj, var_name) is True
        else:
            assert getattr(output_obj, var_name) is False


@pytest.mark.atm
def test_atm_forward_schema_atm_forward_result(srs, capsys):
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


@pytest.mark.atm
def test_atm_forward_custom_spectrum(srs, capsys):
    # init
    request_dict = {
        "request": {
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "output": {
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "height_integrated_rayleighs_lbh": True,
                "height_integrated_rayleighs_1304": True,
                "height_integrated_rayleighs_1356": True,
            },
            "no_cache": True,
        }
    }

    # set custom spectrum
    #
    # Below is taken from crib sheet
    ef_count = 11  # number of energy and flux values we want to have
    custom_spectrum_arr = np.zeros((2, ef_count), order='F', dtype=np.single)  # 2-D array, first dimension will be energy, second will be flux
    for i in range(ef_count):
        custom_spectrum_arr[0, i] = 4000. + i * 100.
        custom_spectrum_arr[1, i] = 1e6
    request_dict["request"]["custom_spectrum"] = custom_spectrum_arr

    # do a request
    result, _ = __do_function(srs, request_dict)

    # check pretty_print method
    result.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.atm
def test_atm_forward_custom_neutral_profile(srs, capsys):
    # init
    request_dict = {
        "request": {
            "timestamp": "2021-11-04T06:00:00",
            "geodetic_latitude": 58.227808,
            "geodetic_longitude": -103.680631,
            "maxwellian_energy_flux": 10,
            "gaussian_energy_flux": 0,
            "maxwellian_characteristic_energy": 5000,
            "gaussian_peak_energy": 1000,
            "gaussian_spectral_width": 200,
            "nrlmsis_model_version": "2.0",
            "oxygen_correction_factor": 1,
            "timescale_auroral": 300,
            "timescale_transport": 300,
            "output": {
                "height_integrated_rayleighs_4278": True,
                "height_integrated_rayleighs_5577": True,
                "height_integrated_rayleighs_6300": True,
                "height_integrated_rayleighs_8446": True,
                "height_integrated_rayleighs_lbh": True,
                "height_integrated_rayleighs_1304": True,
                "height_integrated_rayleighs_1356": True,
            },
            "no_cache": True,
        }
    }

    # set custom neutral profile
    #
    # Below is taken from crib sheet
    #
    # Input is a 7 x num_neut array
    #   - The first row is the altitude (kilometers)
    #   - The 2nd to 6th rows are the densities of O, O2, N2, N, NO (cm^-3)
    #   - The 7th row is the neutral temperature (Kelvin)
    num_neut = 16
    custom_neutral_profile_arr = np.zeros((7, num_neut), order='F', dtype=np.single)
    for i in range(num_neut):
        custom_neutral_profile_arr[0, i] = 50. + i * 50.
        custom_neutral_profile_arr[2, i] = 1e16 * np.exp(-2. * i)
        custom_neutral_profile_arr[3, i] = 2.5e15 * np.exp(-2. * i)
        custom_neutral_profile_arr[4, i] = 1e6
        custom_neutral_profile_arr[5, i] = 1e6
        custom_neutral_profile_arr[6, i] = 200. + i * 50.

        if i < 1:
            custom_neutral_profile_arr[1, i] = 1e9
        else:
            custom_neutral_profile_arr[1, i] = 1e10 * np.exp(-1. * (i - 1.))
    request_dict["request"]["custom_neutral_profile"] = custom_neutral_profile_arr

    # do a request
    result, _ = __do_function(srs, request_dict)

    # check pretty_print method
    result.pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""
