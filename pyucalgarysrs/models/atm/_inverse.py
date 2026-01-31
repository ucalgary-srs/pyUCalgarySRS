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
from .classes_inverse import ATMInverseResult, ATMInverseResultRequestInfo, ATMInverseRequest
from ...exceptions import SRSAPIError


def inverse(
    srs_obj,
    timestamp,
    geodetic_latitude,
    geodetic_longitude,
    intensity_4278,
    intensity_5577,
    intensity_6300,
    intensity_8446,
    output,
    precipitation_flux_spectral_type,
    nrlmsis_model_version,
    special_logic_keyword,
    atm_model_version,
    no_cache,
    timeout,
):
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
        special_logic_keyword=special_logic_keyword,
        output=output,
        no_cache=no_cache,
    )

    # set up request
    url = "%s/api/v2/atm/inverse" % (srs_obj.api_base_url)
    post_data = {
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "geodetic_latitude": geodetic_latitude,
        "geodetic_longitude": geodetic_longitude,
        "intensity_4278": intensity_4278,
        "intensity_5577": intensity_5577,
        "intensity_6300": intensity_6300,
        "intensity_8446": intensity_8446,
        "precipitation_flux_spectral_type": precipitation_flux_spectral_type,
        "nrlmsis_model_version": nrlmsis_model_version,
        "special_logic_keyword": special_logic_keyword,
        "output": output.__dict__,
        "no_cache": no_cache,
    }

    # make request
    try:
        r = requests.post(url, json=post_data, headers=srs_obj.api_headers, timeout=timeout)
    except Exception as e:  # pragma: nocover-ok
        raise SRSAPIError("Unexpected API error: %s" % (str(e))) from e
    if (r.status_code != 200):  # pragma: nocover-ok
        try:
            res = r.json()
            msg = res["detail"]
        except Exception:
            msg = r.content
        raise SRSAPIError("API error code %d: %s" % (r.status_code, msg))
    res = r.json()

    # set up return object
    request_info_obj = ATMInverseResultRequestInfo(
        request=request_obj,
        calculation_duration_ms=res["calculation_duration_ms"],
    )

    # set main result
    result_obj = ATMInverseResult(
        request_info=request_info_obj,
        energy_flux=res["data"]["energy_flux"],
        mean_energy=res["data"]["mean_energy"],
        oxygen_correction_factor=res["data"]["oxygen_correction_factor"],
    )

    # return
    return result_obj
