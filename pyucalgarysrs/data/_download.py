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

import os
import requests
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm.auto import tqdm
from .classes import FileListingResponse, FileDownloadResult, Dataset
from ..exceptions import SRSAPIError, SRSDownloadError


def __download_url(
    url,
    prefix,
    output_base_path,
    headers,
    timeout,
    overwrite=False,
    pbar=None,
    pbar_iterator_nfiles=False,
):
    # set output filename
    output_filename = Path(output_base_path) / Path(url.removeprefix(prefix + "/"))
    if (overwrite is False and os.path.exists(output_filename)):
        if (pbar is not None):
            if (pbar_iterator_nfiles is True):
                pbar.update()
            else:
                pbar.update(os.path.getsize(output_filename))
        return {"filename": output_filename, "bytes_downloaded": 0}

    # create destination directory
    try:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    except Exception:  # pragma: nocover
        # NOTE: sometimes when making directories in parallel there are race conditions. We put
        # in a catch here and carry on if there are ever issues.
        pass

    # retrieve file and save to disk
    r = requests.get(url, headers=headers, timeout=timeout)
    if (r.status_code == 200):
        this_bytes = len(r.content)
        with open(output_filename, 'wb') as fp:
            fp.write(r.content)

        # advance progress
        if (pbar is not None):
            if (pbar_iterator_nfiles is True):
                pbar.update()
            else:
                pbar.update(this_bytes)
    else:
        if (pbar is not None):
            if (pbar_iterator_nfiles is True):
                pbar.update()
            else:
                pbar.update(0)
        raise SRSDownloadError("HTTP error %d when downloading '%s'" % (r.status_code, url))

    # return filename
    return {"filename": output_filename, "bytes_downloaded": this_bytes}


def __download_urls(srs_obj,
                    file_listing_obj,
                    n_parallel,
                    overwrite,
                    progress_bar_disable,
                    progress_bar_ncols,
                    progress_bar_ascii,
                    progress_bar_desc,
                    timeout,
                    progress_bar_format_numurls_nobytes=False):

    def __do_parallel_work(pbar=None, pbar_iterator_nfiles=False):

        def download_task(i):
            return __download_url(
                file_listing_obj.urls[i],
                path_prefix,
                output_path,
                srs_obj.api_headers,
                timeout,
                overwrite=overwrite,
                pbar=pbar,
                pbar_iterator_nfiles=pbar_iterator_nfiles,
            )

        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(download_task, i): i for i in range(len(file_listing_obj.urls))}
            job_data = []
            for future in as_completed(futures):
                job_data.append(future.result())

        return job_data

    # set timeout
    if (timeout is None):
        timeout = srs_obj.api_timeout

    # set output path
    output_path = "%s/%s" % (srs_obj.download_output_root_path, file_listing_obj.dataset.name)

    # set path prefix
    path_prefix = file_listing_obj.path_prefix

    # set progress bar description text
    desc_str = "Downloading %s files" % (file_listing_obj.dataset.name)
    if (progress_bar_desc is not None):
        desc_str = progress_bar_desc

    # download the urls
    parallel_data = []
    if (progress_bar_disable is True):
        parallel_data = __do_parallel_work()
    else:
        if (progress_bar_format_numurls_nobytes is True):
            # total bytes could be inaccurate, progress bar should use count of urls
            # and iterator of 'files' instead of bytes.
            with tqdm(total=len(file_listing_obj.urls), desc=desc_str, unit="files", ncols=progress_bar_ncols, ascii=progress_bar_ascii) as pbar:
                parallel_data = __do_parallel_work(pbar=pbar, pbar_iterator_nfiles=True)
        else:
            # total bytes is accurate, show MB/s
            with tqdm(total=file_listing_obj.total_bytes,
                      desc=desc_str,
                      ncols=progress_bar_ncols,
                      ascii=progress_bar_ascii,
                      unit="B",
                      unit_scale=True) as pbar:
                parallel_data = __do_parallel_work(pbar=pbar)

    # cast into return obj
    filenames_list = []
    total_bytes_downloaded = 0
    for p in parallel_data:
        filenames_list.append(p["filename"])  # type: ignore
        total_bytes_downloaded += p["bytes_downloaded"]  # type: ignore
    download_obj = FileDownloadResult(
        filenames=filenames_list,
        count=len(parallel_data),
        dataset=file_listing_obj.dataset,
        total_bytes=total_bytes_downloaded,
        output_root_path=output_path,
    )

    # return
    return download_obj


def get_urls(srs_obj, dataset_name, start, end, site_uid, device_uid, timeout, warning_stack_level=3):
    # set timeout
    if (timeout is None):
        timeout = srs_obj.api_timeout

    # set up API file listing request
    params = {
        "name": dataset_name,
        "start": start,
        "end": end,
        "include_total_bytes": True,
    }
    if (site_uid is not None):
        params["site_uid"] = site_uid
    if (device_uid is not None):
        params["device_uid"] = device_uid

    # check warnings about site and device being supplied
    if (site_uid is not None and "calibration" in dataset_name.lower()):
        warnings.warn("The site_uid filter was used when retrieving URLs for the dataset %s. Please note "
                      "that the site_uid field is not used when filtering in this dataset. Consider removing "
                      "it from your function call." % (dataset_name),
                      UserWarning,
                      stacklevel=warning_stack_level)
    if (device_uid is not None and "skymap" in dataset_name.lower()):
        warnings.warn("The device_uid filter was used when retrieving URLs for the dataset %s. Please note "
                      "that the device_uid field is not used when filtering in this dataset. Consider removing "
                      "it from your function call." % (dataset_name),
                      UserWarning,
                      stacklevel=warning_stack_level)

    # make API request
    url = "%s/api/v1/data_distribution/urls" % (srs_obj.api_base_url)
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

    # cast response into FileDownloadResult object
    file_listing_obj = FileListingResponse(**res)

    # cast dataset part of response
    file_reading_supported = True if res["dataset"]["name"] in file_reading_supported_datasets else False
    file_listing_obj.dataset = Dataset(**res["dataset"], file_reading_supported=file_reading_supported)

    # return
    return file_listing_obj


def download_generic(srs_obj, dataset_name, start, end, site_uid, device_uid, n_parallel, overwrite, progress_bar_disable, progress_bar_ncols,
                     progress_bar_ascii, progress_bar_desc, timeout):
    # get file listing
    file_listing_obj = get_urls(srs_obj, dataset_name, start, end, site_uid, device_uid, timeout, warning_stack_level=4)

    # check to see if there are files to download
    if (len(file_listing_obj.urls) == 0):
        warnings.warn("No data found to download", stacklevel=1)

    # download the urls
    download_obj = __download_urls(
        srs_obj,
        file_listing_obj,
        n_parallel,
        overwrite,
        progress_bar_disable,
        progress_bar_ncols,
        progress_bar_ascii,
        progress_bar_desc,
        timeout,
    )

    # return
    return download_obj


def download_using_urls(srs_obj, file_listing_obj, n_parallel, overwrite, progress_bar_disable, progress_bar_ncols, progress_bar_ascii,
                        progress_bar_desc, timeout):
    # download the urls
    download_obj = __download_urls(srs_obj,
                                   file_listing_obj,
                                   n_parallel,
                                   overwrite,
                                   progress_bar_disable,
                                   progress_bar_ncols,
                                   progress_bar_ascii,
                                   progress_bar_desc,
                                   timeout,
                                   progress_bar_format_numurls_nobytes=True)

    # return
    return download_obj
