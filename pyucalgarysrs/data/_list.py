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

    # cast response into dataset objects
    datasets = [Dataset(**ds) for ds in res]

    # return
    return datasets
