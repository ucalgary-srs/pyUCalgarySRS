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
from .classes_forward import ATMForwardResult, ATMForwardResultRequestInfo, ATMForwardRequest
from ...exceptions import SRSAPIError


def forward(srs_obj, timestamp, geodetic_latitude, geodetic_longitude, output, maxwellian_energy_flux, gaussian_energy_flux,
            maxwellian_characteristic_energy, gaussian_peak_energy, gaussian_spectral_width, nrlmsis_model_version, oxygen_correction_factor,
            timescale_auroral, timescale_transport, atm_model_version, custom_spectrum, no_cache, timeout):
    # set timeout
    if (timeout is None):
        timeout = srs_obj.api_timeout

    # cast into request object
    request_obj = ATMForwardRequest(
        atm_model_version=atm_model_version,
        timestamp=timestamp,
        geodetic_latitude=geodetic_latitude,
        geodetic_longitude=geodetic_longitude,
        maxwellian_energy_flux=maxwellian_energy_flux,
        gaussian_energy_flux=gaussian_energy_flux,
        maxwellian_characteristic_energy=maxwellian_characteristic_energy,
        gaussian_peak_energy=gaussian_peak_energy,
        gaussian_spectral_width=gaussian_spectral_width,
        nrlmsis_model_version=nrlmsis_model_version,
        oxygen_correction_factor=oxygen_correction_factor,
        timescale_auroral=timescale_auroral,
        timescale_transport=timescale_transport,
        output=output,
        no_cache=no_cache,
        custom_spectrum=custom_spectrum,
    )

    # set up request
    post_data = {
        "atm_model_version": atm_model_version,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "geodetic_latitude": geodetic_latitude,
        "geodetic_longitude": geodetic_longitude,
        "maxwellian_energy_flux": maxwellian_energy_flux,
        "gaussian_energy_flux": gaussian_energy_flux,
        "maxwellian_characteristic_energy": maxwellian_characteristic_energy,
        "gaussian_peak_energy": gaussian_peak_energy,
        "gaussian_spectral_width": gaussian_spectral_width,
        "nrlmsis_model_version": nrlmsis_model_version,
        "oxygen_correction_factor": oxygen_correction_factor,
        "timescale_auroral": timescale_auroral,
        "timescale_transport": timescale_transport,
        "output": output.__dict__,
        "no_cache": no_cache,
    }

    # inject custom spectrum if supplied
    if (custom_spectrum is not None):
        post_data["custom_spectrum"] = {
            "energy": custom_spectrum[:, 0].tolist(),
            "flux": custom_spectrum[:, 1].tolist(),
        }

    # make request
    try:
        url = "%s/api/v1/atm/forward" % (srs_obj.api_base_url)
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
    request_info_obj = ATMForwardResultRequestInfo(request=request_obj, calculation_duration_ms=res["calculation_duration_ms"])
    result_obj = ATMForwardResult(
        request_info=request_info_obj,
        height_integrated_rayleighs_4278=res["data"]["height_integrated_rayleighs_4278"],
        height_integrated_rayleighs_5577=res["data"]["height_integrated_rayleighs_5577"],
        height_integrated_rayleighs_6300=res["data"]["height_integrated_rayleighs_6300"],
        height_integrated_rayleighs_8446=res["data"]["height_integrated_rayleighs_8446"],
        height_integrated_rayleighs_lbh=res["data"]["height_integrated_rayleighs_lbh"],
        height_integrated_rayleighs_1304=res["data"]["height_integrated_rayleighs_1304"],
        height_integrated_rayleighs_1356=res["data"]["height_integrated_rayleighs_1356"],
        altitudes=np.asarray(res["data"]["altitudes"]) if res["data"]["altitudes"] is not None else None,
        emission_4278=np.asarray(res["data"]["emission_4278"]) if res["data"]["emission_4278"] is not None else None,
        emission_5577=np.asarray(res["data"]["emission_5577"]) if res["data"]["emission_5577"] is not None else None,
        emission_6300=np.asarray(res["data"]["emission_6300"]) if res["data"]["emission_6300"] is not None else None,
        emission_8446=np.asarray(res["data"]["emission_8446"]) if res["data"]["emission_8446"] is not None else None,
        emission_lbh=np.asarray(res["data"]["emission_lbh"]) if res["data"]["emission_lbh"] is not None else None,
        emission_1304=np.asarray(res["data"]["emission_1304"]) if res["data"]["emission_1304"] is not None else None,
        emission_1356=np.asarray(res["data"]["emission_1356"]) if res["data"]["emission_1356"] is not None else None,
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
