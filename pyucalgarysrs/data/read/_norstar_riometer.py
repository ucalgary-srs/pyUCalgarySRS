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
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from ..classes import RiometerData

# globals
RIOMETER_DT = np.dtype("float32")
NORSTAR_RIOMETER_3_LETTER_SITE_CODES = {
    "chur": ["chu"],
    "cont": ["con"],
    "daws": ["daw"],
    "arvi": ["esk"],
    "fsim": ["sim"],
    "fsmi": ["smi"],
    "gill": ["gil"],
    "isll": ["isl"],
    "mcmu": ["mcm"],
    "pina": ["pin"],
    "rabb": ["rab"],
    "rank": ["ran"],
    "talo": ["tal"],
}  # NOTE: pulled from UCalgary SRS API code; if you update this, we should update the API too.


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
    rio_data = []
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

        # riometer data
        rio_data.append(RiometerData(timestamp=result["np_timestamp"], raw_signal=result["np_raw_signal"], absorption=result["np_absorption"]))

        # timestamp
        if (len(result["np_timestamp"]) > 0):
            top_level_timestamps.append(result["np_timestamp"][0])

        # metadata
        if (no_metadata is False):
            metadata.append(result["metadata"])

    # return
    return rio_data, top_level_timestamps, metadata, problematic_file_list


def __riometer_readfile_worker(file, no_metadata=False, start_time=None, end_time=None, quiet=False):
    # init
    metadata_dict = {}
    np_timestamp = np.array([], dtype=datetime.datetime)
    np_raw_signal = np.array([], dtype=RIOMETER_DT)
    np_absorption = np.array([], dtype=RIOMETER_DT)
    problematic = False
    error_message = ""

    # determine number of expected columns
    #
    # NOTE: we use the filename here to tell us if it's a k0 or k2 file, since
    # we cannot assume the dataset name will be supplied to the parent functions.
    file_type = None
    if ("_k0_" in file.name or "v0.txt" in file.name):
        file_type = "k0"
        np_absorption = None
    elif ("_k2_" in file.name or "v1a.txt" in file.name):
        file_type = "k2"

    # check file type
    if (file_type is None):
        if (quiet is False):
            print("Error reading file, unknown file type for '%s'" % (file))
        problematic = True
        error_message = "error reading file, unknown file type"
        return {
            "np_timestamp": np_timestamp,
            "np_raw_signal": np_raw_signal,
            "np_absorption": np_absorption,
            "metadata": metadata_dict,
            "problematic": problematic,
            "file": file,
            "error_message": error_message,
        }

    # read the file using numpy
    if (file_type == "k0"):
        # k0 data, 3 columns
        np_date, np_time, np_raw_signal = np.genfromtxt(file, comments='#', dtype="S8,S8,f", unpack=True)
    else:
        # k2 data, 4 columns
        np_date, np_time, np_absorption, np_raw_signal = np.genfromtxt(file, comments='#', dtype="S8,S8,f,f", unpack=True)

    # read the metadata
    if (no_metadata is False):
        try:
            # init
            found_site_uid = None

            # read file's metadata
            fp = open(file, 'r')
            for line in fp:
                # check if we want this line
                if (line[0] != '#'):
                    break
                if ("------------" in line):
                    # end of metadata, bail out
                    break
                line = line.strip()[1:]  # remove the whitespace, remove the # leading character

                # split the line based
                #
                # NOTE: some lines are differently formatted, so we
                # need a few special cases to handle them
                if ("Version" in line):  # version line
                    metadata_dict["version"] = line.strip()
                elif ("----" in line):  # first line
                    metadata_dict["summary"] = line.strip()
                else:
                    line_split = [x.strip() for x in line.split(':', maxsplit=1)]
                    if ("processing date" in line.lower()):  # convert to ISO datetime format
                        metadata_dict["processing_date"] = datetime.datetime.strptime(line_split[1],
                                                                                      "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d %H:%M:%S")
                    elif ("site unique id" in line.lower()):  # site ID line
                        # skip this line since it is not consistently in the metadata, so
                        # we will derive it from the filename and insert it after this for-loop
                        found_site_uid = line_split[1].lower()
                        continue
                    else:
                        metadata_dict[line_split[0].lower().replace(' ', '_')] = line_split[1]
            fp.close()

            # add in the site unique ID if it isn't in there
            if (found_site_uid is not None):  # already have it
                metadata_dict["site_unique_id"] = found_site_uid
            else:
                if ('_' == file.name[3]):
                    # 3-letter site code
                    for s4, s3 in NORSTAR_RIOMETER_3_LETTER_SITE_CODES.items():
                        if (file.name[0:3] in s3):
                            metadata_dict["site_unique_id"] = s4
                            break
                else:
                    # 4-letter site code -- find position of 'rio-' in the file, site code
                    # will be right after that
                    idx = file.name.find('rio-')
                    if (idx != -1):
                        metadata_dict["site_unique_id"] = file[idx + 4:4].lower()
            if ("site_unique_id" not in metadata_dict):
                # still haven't found it, raise a warning message
                metadata_dict["site_unique_id"] = "unknown"
                if (quiet is False):
                    print("Warning: unable to determine 'site_unique_id' field in file '%s'" % (file))
        except Exception as e:
            # set error message
            if (quiet is False):
                print("Error reading metadata for file '%s': %s" % (file, str(e)))
            problematic = True
            error_message = "error reading metadata: %s" % (str(e))

            # reset objects
            metadata_dict = {}
            np_raw_signal = np.array([], dtype=RIOMETER_DT)
            np_absorption = None if file_type == "k0" else np.array([], dtype=RIOMETER_DT)

            # return
            return {
                "np_timestamp": np_timestamp,
                "np_raw_signal": np_raw_signal,
                "np_absorption": np_absorption,
                "metadata": metadata_dict,
                "problematic": problematic,
                "file": file,
                "error_message": error_message,
            }

    # set timestamp numpy array
    try:
        # initialize new object, populate using the date and time arrays
        #
        # NOTE: there can be UT24 records in the files, so we need to keep track
        # of those, and delete them from the raw_signal and absorption arrays
        #
        # NOTE: this is also where we filter based on the start and end times
        idxs_to_delete = []
        np_timestamp = np.empty(np_date.shape, dtype=datetime.datetime)
        for i in range(0, np_date.shape[0]):
            # mark the 24th hour items as indexes to delete
            if (np_time[i][0:2].decode() == "24"):
                idxs_to_delete.append(i)
                continue

            # mark times outside of the valid start and end ranges as indexes to delete
            np_timestamp[i] = datetime.datetime.strptime("%s %s" % (np_date[i].decode(), np_time[i].decode()), "%d/%m/%y %H:%M:%S")
            if ((start_time is None or np_timestamp[i] >= start_time) and (end_time is None or np_timestamp[i] <= end_time)):
                pass
            else:
                idxs_to_delete.append(i)

        # delete rows as needed
        if (len(idxs_to_delete) > 0):
            np_timestamp = np.delete(np_timestamp, idxs_to_delete)
            np_raw_signal = np.delete(np_raw_signal, idxs_to_delete)
            if (np_absorption is not None):
                np_absorption = np.delete(np_absorption, idxs_to_delete)
    except Exception as e:
        if (quiet is False):
            print("Error processing timestamps for file '%s': %s" % (file, str(e)))
        problematic = True
        error_message = "error processing timestamps: %s" % (str(e))

        # set
        metadata_dict = {}
        np_timestamp = np.array([], dtype=datetime.datetime)
        np_raw_signal = np.array([], dtype=RIOMETER_DT)
        np_absorption = None if file_type == "k0" else np.array([], dtype=RIOMETER_DT)

        # return
        return {
            "np_timestamp": np_timestamp,
            "np_raw_signal": np_raw_signal,
            "np_absorption": np_absorption,
            "metadata": metadata_dict,
            "problematic": problematic,
            "file": file,
            "error_message": error_message,
        }

    # return
    return {
        "np_timestamp": np_timestamp,
        "np_raw_signal": np_raw_signal,
        "np_absorption": np_absorption,
        "metadata": metadata_dict,
        "problematic": problematic,
        "file": file,
        "error_message": error_message,
    }
