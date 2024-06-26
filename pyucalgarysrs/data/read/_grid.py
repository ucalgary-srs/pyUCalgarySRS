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

import signal
import h5py
import numpy as np
from pathlib import Path
from multiprocessing import Pool


def read(file_list, n_parallel=1, first_record=False, no_metadata=False, tar_tempdir=None, quiet=False):
    # if input is just a single file name in a string, convert to a list to be fed to the workers
    if isinstance(file_list, str) or isinstance(file_list, Path):
        file_list = [file_list]

    # convert to object, injecting other data we need for processing
    processing_list = []
    for f in file_list:
        processing_list.append({
            "filename": str(f),
            "first_record": first_record,
            "no_metadata": no_metadata,
            "quiet": quiet,
        })

    # check n_parallel
    if (n_parallel > 1):
        try:
            # set up process pool (ignore SIGINT before spawning pool so child processes inherit SIGINT handler)
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = Pool(processes=n_parallel)
            signal.signal(signal.SIGINT, original_sigint_handler)  # restore SIGINT handler
        except ValueError:  # pragma: nocover
            # likely the read call is being used within a context that doesn't support the usage
            # of signals in this way, proceed without it
            pool = Pool(processes=n_parallel)

        # call readfile function, run each iteration with a single input file from file_list
        # NOTE: structure of data - data[file][metadata dictionary lists = 1, images = 0][frame]
        pool_data = []
        try:
            pool_data = pool.map(__grid_readfile_worker, processing_list)
        except KeyboardInterrupt:  # pragma: nocover
            pool.terminate()  # gracefully kill children
            return np.empty((0, 0, 0, 0)), [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        pool_data = []
        for p in processing_list:
            pool_data.append(__grid_readfile_worker(p))

    # pre-allocate data arrays
    data_dict = {}
    data_dict_sizes_and_dtype = {}
    for pd in pool_data:
        # process grid
        if ("grid" not in data_dict_sizes_and_dtype.keys()):
            data_dict_sizes_and_dtype["grid"] = {
                "shape": list(pd["data"]["grid"].shape),
                "dtype": pd["data"]["grid"].dtype,
                "n_channels": 1,
            }
            if (len(data_dict_sizes_and_dtype["grid"]["shape"]) == 4):
                data_dict_sizes_and_dtype["grid"]["n_channels"] = 3
        else:
            data_dict_sizes_and_dtype["grid"]["shape"][-1] += list(pd["data"]["grid"].shape)[-1]

        # process timestamp
        if ("timestamp" not in data_dict_sizes_and_dtype.keys()):
            data_dict_sizes_and_dtype["timestamp"] = {
                "shape": list(pd["data"]["timestamp"].shape),
                "dtype": pd["data"]["timestamp"].dtype,
                "n_channels": 1,
            }
            if (len(data_dict_sizes_and_dtype["timestamp"]["shape"]) == 4):
                data_dict_sizes_and_dtype["timestamp"]["n_channels"] = 3
        else:
            data_dict_sizes_and_dtype["timestamp"]["shape"][-1] += list(pd["data"]["timestamp"].shape)[-1]

        # process all source type
        if ("source_info" in pd["data"].keys()):
            if ("source_info" not in data_dict_sizes_and_dtype):
                data_dict_sizes_and_dtype["source_info"] = {}
            for key_name in pd["data"]["source_info"].keys():
                if (key_name not in data_dict_sizes_and_dtype["source_info"].keys()):
                    data_dict_sizes_and_dtype["source_info"][key_name] = {
                        "shape": list(pd["data"]["source_info"][key_name].shape),
                        "dtype": pd["data"]["source_info"][key_name].dtype,
                        "n_channels": 1,
                    }
                    if (len(data_dict_sizes_and_dtype["source_info"][key_name]["shape"]) == 4):
                        data_dict_sizes_and_dtype["source_info"][key_name]["n_channels"] = 3
                else:
                    data_dict_sizes_and_dtype["source_info"][key_name]["shape"][-1] += list(pd["data"]["source_info"][key_name].shape)[-1]
    for key, value in data_dict_sizes_and_dtype.items():
        if (key == "source_info"):
            if ("source_info" not in data_dict.keys()):
                data_dict["source_info"] = {}
            for source_info_key, _ in value.items():
                data_dict["source_info"][source_info_key] = np.empty(
                    tuple(data_dict_sizes_and_dtype["source_info"][source_info_key]["shape"]),
                    dtype=data_dict_sizes_and_dtype["source_info"][source_info_key]["dtype"],
                )
        else:
            data_dict[key] = np.empty(
                tuple(data_dict_sizes_and_dtype[key]["shape"]),
                dtype=data_dict_sizes_and_dtype[key]["dtype"],
            )

    # pre-allocate metadata list
    metadata_dict_list = [{}] * data_dict_sizes_and_dtype["timestamp"]["shape"][0]
    problematic_file_list = []

    # for each pool result
    list_position = 0
    for pd in pool_data:
        # populate data
        for key, value in data_dict.items():
            this_end_position = None
            if (key == "source_info"):
                for key1, _ in value.items():
                    this_end_position = list_position + list(pd["data"]["source_info"][key1].shape)[-1]
                    # print("source_info", key1, list_position, this_end_position)
                    if (data_dict_sizes_and_dtype["source_info"][key1]["n_channels"] == 1):
                        data_dict["source_info"][key1][:, :, list_position:this_end_position] = pd["data"]["source_info"][key1][:, :, :]
                    else:
                        data_dict["source_info"][key1][:, :, :, list_position:this_end_position] = pd["data"]["source_info"][key1][:, :, :, :]
            elif (key == "timestamp"):
                this_end_position = list_position + list(pd["data"][key].shape)[-1]
                # print(key, list_position, this_end_position)
                data_dict[key][list_position:this_end_position] = pd["data"][key][:]
            else:
                this_end_position = list_position + list(pd["data"][key].shape)[-1]
                # print(key, list_position, this_end_position)
                if (data_dict_sizes_and_dtype[key]["n_channels"] == 1):
                    data_dict[key][:, :, list_position:this_end_position] = pd["data"][key][:, :, :]
                else:
                    data_dict[key][:, :, :, list_position:this_end_position] = pd["data"][key][:, :, :, :]
        list_position = this_end_position  # type: ignore

        # populate metadata
        if (no_metadata is False):
            metadata_dict_list = pd["metadata"]

    # return
    pool_data = None
    return data_dict, metadata_dict_list, problematic_file_list


def __grid_readfile_worker(file_obj):
    # init
    data_dict = {}
    metadata_dict_list = []
    problematic = False
    error_message = ""

    # open H5 file
    f = h5py.File(file_obj["filename"], 'r')

    # extract the grid
    data_dict["grid"] = f["data"]["grid"][:]  # type: ignore

    # flip data (since it's upside down with displaying bottom-up (imshow origin="bottom"))
    data_dict["grid"] = np.flip(data_dict["grid"], axis=0)  # type: ignore

    # extract the timestamp
    data_dict["timestamp"] = f["data"]["timestamp"][:]  # type: ignore

    # extract source info data
    data_dict["source_info"] = {}
    source_info_keys = f["data"]["source_info"].keys()  # type: ignore
    if (len(source_info_keys) > 0):
        for key_name in source_info_keys:
            data_dict["source_info"][key_name] = f["data"]["source_info"][key_name][:]  # type: ignore

    # extract metadata
    file_metadata = {}
    if (file_obj["no_metadata"] is True):
        metadata_dict_list = [{}] * len(data_dict["timestamp"])  # type: ignore
    else:
        # get file metadata
        for key, value in f["metadata"]["file"].attrs.items():  # type: ignore
            file_metadata[key] = value

        # read frame metadata
        for i in range(0, len(data_dict["timestamp"])):  # type: ignore
            this_frame_metadata = file_metadata.copy()
            for key, value in f["metadata"]["frame"]["frame%d" % (i)].attrs.items():  # type: ignore
                this_frame_metadata[key] = value
            metadata_dict_list.append(this_frame_metadata)

    # close file
    f.close()

    # return
    return {
        "data": data_dict,
        "metadata": metadata_dict_list,
        "problematic": problematic,
        "filename": file_obj["filename"],
        "error_message": error_message,
    }
