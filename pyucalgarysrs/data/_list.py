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
from .classes import Dataset, Observatory
from ..exceptions import SRSAPIError


def list_datasets(srs_obj, name, timeout, supported_library):
    # set timeout
    if (timeout is None):
        timeout = srs_obj.api_timeout

    # set up request
    params = {}
    if (name != ""):
        params["name"] = name

    # make request
    url = "%s/api/v1/data_distribution/datasets" % (srs_obj.api_base_url)
    try:
        r = requests.get(url, params=params, headers=srs_obj.api_headers, timeout=timeout)
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

    # get list of file reading supported datasets
    file_reading_supported_datasets = srs_obj.data.list_supported_read_datasets()

    # cast response into dataset objects
    datasets = res
    for i in range(0, len(datasets)):
        # inject 'data_reading_supported' flag
        datasets[i]["file_reading_supported"] = True if datasets[i]["name"] in file_reading_supported_datasets else False

        # cast into object
        datasets[i] = Dataset(**datasets[i])

    # filter out any based on the 'supported_libraries' argument
    filtered_datasets = []
    for d in datasets:
        if (supported_library in d.supported_libraries):
            filtered_datasets.append(d)

    # return
    return filtered_datasets


def list_observatories(srs_obj, instrument_array, uid, timeout):
    # set timeout
    if (timeout is None):
        timeout = srs_obj.api_timeout

    # set up request
    params = {"instrument_array": instrument_array}
    if (uid is not None):
        params["uid"] = uid

    # make request
    url = "%s/api/v1/data_distribution/observatories" % (srs_obj.api_base_url)
    try:
        r = requests.get(url, params=params, headers=srs_obj.api_headers, timeout=timeout)
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

    # cast response into observatory objects
    sites = [Observatory(**x) for x in res]

    # return
    return sites
