import os
import requests
import joblib
from tqdm import tqdm, tqdm_notebook
from ..exceptions import SRSAPIException
from ._schemas import FileListingResponse, FileDownloadResult, Dataset


def __download_url(url, prefix, output_base_path, overwrite=False, pbar=None, pbar_iterator_nfiles=False):
    # set output filename
    output_filename = "%s/%s" % (output_base_path, url.removeprefix(prefix + "/"))
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
    r = requests.get(url)
    this_bytes = len(r.content)
    with open(output_filename, 'wb') as fp:
        fp.write(r.content)

    # advance progress
    if (pbar is not None):
        if (pbar_iterator_nfiles is True):
            pbar.update()
        else:
            pbar.update(this_bytes)

    # return filename
    return {"filename": output_filename, "bytes_downloaded": this_bytes}


def _download_urls(srs_obj,
                   file_listing_obj,
                   n_parallel,
                   overwrite,
                   progress_bar_disable,
                   progress_bar_ncols,
                   progress_bar_ascii,
                   progress_bar_desc,
                   progress_bar_format_numurls_nobytes=False):

    def __do_parallel_work(pbar=None, pbar_iterator_nfiles=False):
        job_data = joblib.Parallel(n_jobs=n_parallel, prefer="threads")(joblib.delayed(__download_url)(
            file_listing_obj.urls[i],
            path_prefix,
            output_path,
            overwrite=overwrite,
            pbar=pbar,
            pbar_iterator_nfiles=pbar_iterator_nfiles,
        ) for i in range(0, len(file_listing_obj.urls)))
        return list(job_data)

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
        if (srs_obj.in_jupyter_notebook is True):  # pragma: nocover
            if (progress_bar_format_numurls_nobytes is True):
                # total bytes could be inaccurate, progress bar should use count of urls
                # and iterator of 'files' instead of bytes.
                with tqdm_notebook(total=len(file_listing_obj.urls), desc=desc_str, unit="files") as pbar:
                    parallel_data = __do_parallel_work(pbar=pbar, pbar_iterator_nfiles=True)
            else:
                # total bytes is accurate, show MB/s
                with tqdm_notebook(total=file_listing_obj.total_bytes, desc=desc_str, unit="B", unit_scale=True) as pbar:
                    parallel_data = __do_parallel_work(pbar=pbar)
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


def _get_urls(srs_obj, dataset_name, start, end, site_uid):
    # set up API file listing request
    params = {
        "name": dataset_name,
        "start": start,
        "end": end,
        "include_total_bytes": True,
    }
    if (site_uid is not None):
        params["site_uid"] = site_uid

    # make API request
    url = "%s/api/v1/data_distribution/urls" % (srs_obj.api_base_url)
    try:
        r = requests.get(url, params=params)
        res = r.json()
    except Exception as e:  # pragma: nocover
        raise SRSAPIException("Unexpected API error: %s" % (str(e))) from e
    if (r.status_code != 200):  # pragma: nocover
        raise SRSAPIException("API error code %d: %s" % (r.status_code, res["detail"]))

    # get list of file reading supported datasets
    file_reading_supported_datasets = srs_obj.data.list_supported_read_datasets()

    # cast response into FileDownloadResult object
    file_listing_obj = FileListingResponse(**res)

    # cast dataset part of response
    file_reading_supported = True if res["dataset"]["name"] in file_reading_supported_datasets else False
    file_listing_obj.dataset = Dataset(**res["dataset"], file_reading_supported=file_reading_supported)

    # return
    return file_listing_obj


def _download_generic(srs_obj, dataset_name, start, end, site_uid, n_parallel, overwrite, progress_bar_disable, progress_bar_ncols,
                      progress_bar_ascii, progress_bar_desc):
    # get file listing
    file_listing_obj = _get_urls(srs_obj, dataset_name, start, end, site_uid)

    # download the urls
    download_obj = _download_urls(
        srs_obj,
        file_listing_obj,
        n_parallel,
        overwrite,
        progress_bar_disable,
        progress_bar_ncols,
        progress_bar_ascii,
        progress_bar_desc,
    )

    # return
    return download_obj


def _download_using_urls(srs_obj, file_listing_obj, n_parallel, overwrite, progress_bar_disable, progress_bar_ncols, progress_bar_ascii,
                         progress_bar_desc):
    # download the urls
    download_obj = _download_urls(srs_obj,
                                  file_listing_obj,
                                  n_parallel,
                                  overwrite,
                                  progress_bar_disable,
                                  progress_bar_ncols,
                                  progress_bar_ascii,
                                  progress_bar_desc,
                                  progress_bar_format_numurls_nobytes=True)

    # return
    return download_obj