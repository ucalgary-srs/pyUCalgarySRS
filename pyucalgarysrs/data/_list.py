import requests
from ._schemas import Dataset
from ..exceptions import SRSAPIException


def list_datasets(srs_obj, name):
    # init
    datasets = []

    # set up request
    params = {}
    if (name != ""):
        params["name"] = name

    # make request
    url = "%s/api/v1/data_distribution/datasets" % (srs_obj.api_base_url)
    try:
        r = requests.get(url, params=params)
        res = r.json()
    except Exception as e:  # pragma: nocover
        raise SRSAPIException("Unexpected API error: %s" % (str(e))) from e
    if (r.status_code != 200):  # pragma: nocover
        raise SRSAPIException("API error code %d: %s" % (r.status_code, res["detail"]))

    # get list of file reading supported datasets
    file_reading_supported_datasets = srs_obj.data.list_supported_read_datasets()

    # cast response into dataset objects
    datasets = res
    for i in range(0, len(datasets)):
        # inject 'data_reading_supported' flag
        datasets[i]["file_reading_supported"] = True if datasets[i]["name"] in file_reading_supported_datasets else False

        # cast into object
        datasets[i] = Dataset(**datasets[i])

    # return
    return datasets
