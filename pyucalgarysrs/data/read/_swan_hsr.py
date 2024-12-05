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

import datetime
import signal
import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from ..classes import HSRData

# globals
HSR_DT = np.dtype("float32")


def read(file_list, n_parallel=1, no_metadata=False, start_time=None, end_time=None, quiet=False):
    # if input is just a single file name in a string, convert to a list to be fed to the workers
    if isinstance(file_list, str) or isinstance(file_list, Path):
        file_list = [file_list]

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
        data = []
        try:
            data = pool.map(partial(
                __riometer_readfile_worker,
                no_metadata=no_metadata,
                start_time=start_time,
                end_time=end_time,
                quiet=quiet,
            ), file_list)
        except KeyboardInterrupt:  # pragma: nocover
            pool.terminate()  # gracefully kill children
            return [], [], [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        data = []
        for f in file_list:
            data.append(__riometer_readfile_worker(
                f,
                no_metadata=no_metadata,
                start_time=start_time,
                end_time=end_time,
                quiet=quiet,
            ))

    # compile results
    hsr_data = []
    top_level_timestamps = []
    metadata = []
    problematic_file_list = []
    for result in data:
        # problematic
        if (result["problematic"] is True):
            problematic_file_list.append({
                "filename": result["file"],
                "error_message": result["error_message"],
            })
            continue

        # hsr data
        hsr_data.append(
            HSRData(
                band_central_frequency=result["band_central_frequency_list"],
                band_passband=result["band_passband_list"],
                timestamp=result["np_timestamp"],
                raw_power=result["np_raw_power"],
                absorption=result["np_absorption"],
            ))

        # timestamps
        if (len(result["np_timestamp"]) > 0):
            top_level_timestamps.append(result["np_timestamp"][0])

        # metadata
        if (no_metadata is False):
            metadata.append(result["metadata"])

    # return
    return hsr_data, top_level_timestamps, metadata, problematic_file_list


def __str_to_datetime_formatter(timestamp_str):
    return datetime.datetime.strptime(timestamp_str.decode(), "%Y-%m-%d %H:%M:%S UTC")


def __riometer_readfile_worker(file, no_metadata=False, start_time=None, end_time=None, quiet=False):
    # init
    metadata_dict = {}
    band_central_frequency_list = []
    band_passband_list = []
    np_timestamp = np.array([], dtype=datetime.datetime)
    np_raw_power = np.array([], dtype=HSR_DT)
    np_absorption = np.array([], dtype=HSR_DT)
    problematic = False
    error_message = ""

    # process file
    try:
        # open H5 file
        f = h5py.File(file, 'r')

        # get timestamp, convert into datetime objects
        np_timestamp = f["data"]["timestamp"][:]  # type: ignore
        np_timestamp = np.vectorize(__str_to_datetime_formatter)(np_timestamp)  # type: ignore
        np_timestamp = np_timestamp.astype(datetime.datetime)

        # determine indexes to keep based on start and end times
        idxs = []
        for idx, val in enumerate(np_timestamp):
            if ((start_time is None or val >= start_time) and (end_time is None or val <= end_time)):
                idxs.append(idx)
        np_timestamp = np_timestamp[idxs]

        # get raw power, timestamp
        np_raw_power = f["data"]["raw_power"][:, idxs]  # type: ignore

        # get band central frequency, bandpass
        band_central_frequency_list = f["data"]["band_central_frequency"][:].tolist()  # type: ignore
        band_passband_list = f["data"]["band_passband"][:].tolist()  # type: ignore

        # set absorption
        if ("_k0_" in file.name):
            # k0 data, no absorption
            np_absorption = None

        # get metadata
        if (no_metadata is False):
            for key, value in f["metadata"]["file"].attrs.items():  # type: ignore
                metadata_dict[key] = value

        # close file
        f.close()
    except Exception as e:
        # set error message
        if (quiet is False):
            print("Failed to read file '%s': %s" % (file, str(e)))
        problematic = True
        error_message = "failed to open file: %s" % (str(e))

        # clear values
        band_central_frequency_list = []
        band_passband_list = []
        np_timestamp = np.array([], dtype=datetime.datetime)
        np_raw_power = np.array([], dtype=HSR_DT)
        np_absorption = np.array([], dtype=HSR_DT)

        # close file
        try:
            f.close()  # type: ignore
        except Exception:
            pass

        # return
        return {
            "band_central_frequency_list": band_central_frequency_list,
            "band_passband_list": band_passband_list,
            "np_timestamp": np_timestamp,
            "np_raw_power": np_raw_power,
            "np_absorption": np_absorption,
            "metadata": metadata_dict,
            "problematic": problematic,
            "file": file,
            "error_message": error_message,
        }

    # return
    return {
        "band_central_frequency_list": band_central_frequency_list,
        "band_passband_list": band_passband_list,
        "np_timestamp": np_timestamp,
        "np_raw_power": np_raw_power,
        "np_absorption": np_absorption,
        "metadata": metadata_dict,
        "problematic": problematic,
        "file": file,
        "error_message": error_message,
    }
