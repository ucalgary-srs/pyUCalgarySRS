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

import requests
import numpy as np
from .classes_inverse import ATMInverseResult, ATMInverseResultRequestInfo, ATMInverseRequest, ATMInverseForwardParams
from ...exceptions import SRSAPIError


def inverse(srs_obj, timestamp, geodetic_latitude, geodetic_longitude, intensity_4278, intensity_5577, intensity_6300, intensity_8446, output,
            precipitation_flux_spectral_type, nrlmsis_model_version, atmospheric_attenuation_correction, atm_model_version, no_cache, timeout):
    # set timeout
    if (timeout is None):
        timeout = srs_obj.api_timeout

    # cast into request object
    request_obj = ATMInverseRequest(
        atm_model_version=atm_model_version,
        timestamp=timestamp,
        geodetic_latitude=geodetic_latitude,
        geodetic_longitude=geodetic_longitude,
        intensity_4278=intensity_4278,
        intensity_5577=intensity_5577,
        intensity_6300=intensity_6300,
        intensity_8446=intensity_8446,
        precipitation_flux_spectral_type=precipitation_flux_spectral_type,
        nrlmsis_model_version=nrlmsis_model_version,
        atmospheric_attenuation_correction=atmospheric_attenuation_correction,
        output=output,
        no_cache=no_cache,
    )

    # set up request
    post_data = {
        "atm_model_version": atm_model_version,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "geodetic_latitude": geodetic_latitude,
        "geodetic_longitude": geodetic_longitude,
        "intensity_4278": intensity_4278,
        "intensity_5577": intensity_5577,
        "intensity_6300": intensity_6300,
        "intensity_8446": intensity_8446,
        "precipitation_flux_spectral_type": precipitation_flux_spectral_type,
        "nrlmsis_model_version": nrlmsis_model_version,
        "output": output.__dict__,
        "no_cache": no_cache,
    }

    # make request
    try:
        url = "%s/api/v1/atm/inverse" % (srs_obj.api_base_url)
        r = requests.post(url, json=post_data, headers=srs_obj.api_headers, timeout=timeout)
    except Exception as e:  # pragma: nocover
        raise SRSAPIError("Unexpected API error: %s" % (str(e))) from e
    if (r.status_code != 200):  # pragma: nocover
        try:
            res = r.json()
            msg = res["detail"]
        except Exception:
            msg = r.content
        raise SRSAPIError("API error code %d: %s" % (r.status_code, msg))
    res = r.json()

    # cast result into return object
    forward_params_obj = None
    if (res["forward_params"] is not None):
        forward_params_obj = ATMInverseForwardParams(**res["forward_params"])
    request_info_obj = ATMInverseResultRequestInfo(
        request=request_obj,
        forward_params=forward_params_obj,
        calculation_duration_ms=res["calculation_duration_ms"],
    )
    result_obj = ATMInverseResult(
        request_info=request_info_obj,
        energy_flux=res["data"]["energy_flux"],
        characteristic_energy=res["data"]["characteristic_energy"],
        oxygen_correction_factor=res["data"]["oxygen_correction_factor"],
        height_integrated_rayleighs_4278=res["data"]["height_integrated_rayleighs_4278"],
        height_integrated_rayleighs_5577=res["data"]["height_integrated_rayleighs_5577"],
        height_integrated_rayleighs_6300=res["data"]["height_integrated_rayleighs_6300"],
        height_integrated_rayleighs_8446=res["data"]["height_integrated_rayleighs_8446"],
        altitudes=np.asarray(res["data"]["altitudes"]) if res["data"]["altitudes"] is not None else None,
        emission_4278=np.asarray(res["data"]["emission_4278"]) if res["data"]["emission_4278"] is not None else None,
        emission_5577=np.asarray(res["data"]["emission_5577"]) if res["data"]["emission_5577"] is not None else None,
        emission_6300=np.asarray(res["data"]["emission_6300"]) if res["data"]["emission_6300"] is not None else None,
        emission_8446=np.asarray(res["data"]["emission_8446"]) if res["data"]["emission_8446"] is not None else None,
        plasma_electron_density=np.asarray(res["data"]["plasma_electron_density"]) if res["data"]["plasma_electron_density"] is not None else None,
        plasma_o2plus_density=np.asarray(res["data"]["plasma_o2plus_density"]) if res["data"]["plasma_o2plus_density"] is not None else None,
        plasma_noplus_density=np.asarray(res["data"]["plasma_noplus_density"]) if res["data"]["plasma_noplus_density"] is not None else None,
        plasma_oplus_density=np.asarray(res["data"]["plasma_oplus_density"]) if res["data"]["plasma_oplus_density"] is not None else None,
        plasma_ionisation_rate=np.asarray(res["data"]["plasma_ionisation_rate"]) if res["data"]["plasma_ionisation_rate"] is not None else None,
        plasma_electron_temperature=np.asarray(res["data"]["plasma_electron_temperature"])
        if res["data"]["plasma_electron_temperature"] is not None else None,
        plasma_ion_temperature=np.asarray(res["data"]["plasma_ion_temperature"]) if res["data"]["plasma_ion_temperature"] is not None else None,
        plasma_pederson_conductivity=np.asarray(res["data"]["plasma_pederson_conductivity"])
        if res["data"]["plasma_pederson_conductivity"] is not None else None,
        plasma_hall_conductivity=np.asarray(res["data"]["plasma_hall_conductivity"]) if res["data"]["plasma_hall_conductivity"] is not None else None,
        neutral_o2_density=np.asarray(res["data"]["neutral_o2_density"]) if res["data"]["neutral_o2_density"] is not None else None,
        neutral_o_density=np.asarray(res["data"]["neutral_o_density"]) if res["data"]["neutral_o_density"] is not None else None,
        neutral_n2_density=np.asarray(res["data"]["neutral_n2_density"]) if res["data"]["neutral_n2_density"] is not None else None,
        neutral_n_density=np.asarray(res["data"]["neutral_n_density"]) if res["data"]["neutral_n_density"] is not None else None,
        neutral_temperature=np.asarray(res["data"]["neutral_temperature"]) if res["data"]["neutral_temperature"] is not None else None,
    )

    # return
    return result_obj
