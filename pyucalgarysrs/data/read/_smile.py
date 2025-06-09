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
import datetime
import signal
import h5py
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from ..._util import show_warning
from ...exceptions import SRSError

# static globals
__SMILE_H5_DT = np.dtype("uint8")


def read(file_list, n_parallel=1, first_record=False, no_metadata=False, start_time=None, end_time=None, tar_tempdir=None, quiet=False):
    # throw up warning if the no_metadata flag is true and start/end times supplied
    #
    # NOTE: since the timestamp info is in the metadata, we can't filter based on
    # start or end time if we were told to not read the metadata, so we throw up
    # a warning and read all the data, and don't read the metadata (no_metadata being
    # true takes priority).
    if (no_metadata is True and (start_time is not None or end_time is not None)):
        show_warning("Cannot filter on start or end time if the no_metadata parameter is set to True. Set no_metadata to False " +
                     "to allow filtering on times. Will proceed by skipping filtering on times and returning no metadata.",
                     stacklevel=1)
        start_time = None
        end_time = None

    # if input is just a single file name in a string, convert to a list to be fed to the workers
    if isinstance(file_list, str) or isinstance(file_list, Path):
        file_list = [file_list]

    # check if anything in the list
    if (len(file_list) == 0):
        if (quiet is False):
            print("No files found to read")
        return np.empty((0, 0, 0, 0)), [], []

    # convert to object, injecting other data we need for processing
    processing_list = []
    for f in file_list:
        processing_list.append({
            "filename": str(f),
            "tar_tempdir": tar_tempdir,
            "first_record": first_record,
            "no_metadata": no_metadata,
            "start_time": start_time,
            "end_time": end_time,
            "quiet": quiet,
        })

    # check n_parallel
    if (n_parallel > 1):
        try:
            # set up process pool (ignore SIGINT before spawning pool so child processes inherit SIGINT handler)
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = Pool(processes=n_parallel)
            signal.signal(signal.SIGINT, original_sigint_handler)  # restore SIGINT handler
        except ValueError:  # pragma: nocover-ok
            # likely the read call is being used within a context that doesn't support the usage
            # of signals in this way, proceed without it
            pool = Pool(processes=n_parallel)

        # call readfile function, run each iteration with a single input file from file_list
        # NOTE: structure of data - data[file][metadata dictionary lists = 1, images = 0][frame]
        pool_data = []
        try:
            pool_data = pool.map(__trex_readfile_worker, processing_list)
        except KeyboardInterrupt:  # pragma: nocover-ok
            pool.terminate()  # gracefully kill children
            return np.empty((0, 0, 0, 0)), [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        pool_data = []
        for p in processing_list:
            pool_data.append(__trex_readfile_worker(p))

    # set image sizes
    image_width = None
    image_height = None
    image_channels = None
    image_dtype = None
    for i in range(0, len(pool_data)):
        # set sizes
        if (pool_data[i][2] is False):
            # not a problematic file
            image_width = pool_data[i][5] if pool_data[i][5] is not None and pool_data[i][5] != 0 else image_width
            image_height = pool_data[i][6] if pool_data[i][6] is not None and pool_data[i][6] != 0 else image_height
            image_channels = pool_data[i][7] if pool_data[i][7] is not None and pool_data[i][7] != 0 else image_channels
            image_dtype = pool_data[i][8] if pool_data[i][8] is not None and pool_data[i][8] != 0 else image_dtype
        else:
            # a problematic file
            image_width = pool_data[i][5] if image_width is None and pool_data[i][5] is not None else image_width
            image_height = pool_data[i][6] if image_height is None and pool_data[i][6] is not None else image_height
            image_channels = pool_data[i][7] if image_channels is None and pool_data[i][7] is not None else image_channels
            image_dtype = pool_data[i][8] if image_dtype is None and pool_data[i][8] is not None else image_dtype
    if (image_width is None or image_height is None or image_channels is None or image_dtype is None):  # pragma: nocover
        raise SRSError("Unexpected read error, please contact the UCalgary team")

    # set image sizes and derive number of frames to prepare for
    total_num_frames = 0
    for i in range(0, len(pool_data)):
        if (pool_data[i][2] is True):
            continue
        if (len(pool_data[i][0]) != 0):  # type: ignore
            total_num_frames += pool_data[i][0].shape[3]  # type: ignore

    # set tasks
    list_position = 0
    tasks = []
    problematic_file_list = []
    for i in range(0, len(pool_data)):
        # check if file was problematic
        if (pool_data[i][2] is True):
            problematic_file_list.append({
                "filename": pool_data[i][3],
                "error_message": pool_data[i][4],
            })
            continue

        # check if any data was read in
        if (len(pool_data[i][0]) == 0):  # type: ignore
            continue

        # find actual number of frames, this may differ from predicted due to dropped frames, end
        # or start of imaging
        this_num_frames = pool_data[i][0].shape[3]  # type: ignore

        # set task
        tasks.append((list_position, list_position + this_num_frames, i))

        # advance list position
        list_position = list_position + this_num_frames

    # pre-allocate array sizes
    images = np.empty([image_height, image_width, image_channels, list_position], dtype=image_dtype)
    if (no_metadata is False):
        metadata_dict_list = [{}] * list_position
    else:
        metadata_dict_list = []

    # merge data using number of threads
    def assemble_data(slice_idx1, slice_idx2, data_idx):
        # merge metadata
        if (no_metadata is False):
            metadata_dict_list[slice_idx1:slice_idx2] = pool_data[data_idx][1]  # type: ignore

        # merge image data
        images[:, :, :, slice_idx1:slice_idx2] = pool_data[data_idx][0]  # type: ignore

        return data_idx

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        for t in tasks:
            executor.submit(assemble_data, t[0], t[1], t[2])

    # ensure entire array views as the desired dtype
    images = images.astype(__SMILE_H5_DT)

    # return
    pool_data = None
    return images, metadata_dict_list, problematic_file_list


def __trex_readfile_worker(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = __SMILE_H5_DT  # type: ignore

    # check file extension to know how to process
    try:
        if (file_obj["filename"].endswith("h5")):
            return __smile_readfile_worker_h5(file_obj)
        else:
            if (file_obj["quiet"] is False):
                print("Unrecognized file type: %s" % (file_obj["filename"]))
            problematic = True
            error_message = "Unrecognized file type"
    except Exception as e:  # pragma: nocover
        if (file_obj["quiet"] is False):
            print("Failed to process file '%s' " % (file_obj["filename"]))
        problematic = True
        error_message = "failed to process file: %s" % (str(e))
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype


def __smile_readfile_worker_h5(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = __SMILE_H5_DT

    # set start and end times so we can use shorter variable names lower down in this function
    start_time = file_obj["start_time"]
    end_time = file_obj["end_time"]

    # extract start and end times of the filename
    try:
        file_dt = datetime.datetime.strptime(os.path.basename(file_obj["filename"])[0:13], "%Y%m%d_%H%M")
    except Exception:
        if (file_obj["quiet"] is False):
            print("Failed to extract timestamp from filename")
        problematic = True
        error_message = "failed to extract timestamp from filename"
        return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
            image_width, image_height, image_channels, image_dtype

    # cross-check the filename with the start and end times; this will allow
    # files that are outside of the desired time frame to not actually bother
    # with getting read
    if ((start_time is None or file_dt >= start_time.replace(second=0, microsecond=0))
            and (end_time is None or file_dt <= end_time.replace(second=0, microsecond=0))):
        # this file should be read
        pass
    else:
        # this file doesn't need to be read
        return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
            image_width, image_height, image_channels, image_dtype

    # process file
    try:
        # open H5 file
        f = h5py.File(file_obj["filename"], 'r')

        # set image shape vars
        image_height = f["data"]["images"].shape[0]  # type: ignore
        image_width = f["data"]["images"].shape[1]  # type: ignore
        image_channels = f["data"]["images"].shape[2]  # type: ignore

        # get timestamps
        if (file_obj["first_record"] is True):
            timestamps = [f["data"]["timestamp"][0]]  # type: ignore
        else:
            timestamps = f["data"]["timestamp"][:]  # type: ignore

        # search through timestamps for indexes to read data for, based
        # on the start and end times
        idxs = []
        if (start_time is None and end_time is None):
            # no filtering, read all indexes
            idxs = np.arange(0, len(timestamps))  # type: ignore
        else:
            for i, val in enumerate(timestamps):  # type: ignore
                val_dt = datetime.datetime.strptime(val.decode(), "%Y-%m-%d %H:%M:%S.%f UTC").replace(microsecond=0)
                if ((start_time is None or val_dt >= start_time) and (end_time is None or val_dt <= end_time)):
                    # matches start and end time range, we want to read this index
                    idxs.append(i)

        # bail out if we don't want to read any frames
        if (len(idxs) == 0):  # pragma: nocover
            return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
                image_width, image_height, image_channels, image_dtype

        # get images
        images = f["data"]["images"][:, :, :, idxs]  # type: ignore

        # read metadata
        file_metadata = {}
        if (file_obj["no_metadata"] is False):
            # get file metadata
            for key, value in f["metadata"]["file"].attrs.items():  # type: ignore
                file_metadata[key] = value

            # read frame metadata
            for i in idxs:  # type: ignore
                this_frame_metadata = file_metadata.copy()
                for key, value in f["metadata"]["frame"]["frame%d" % (i)].attrs.items():  # type: ignore
                    this_frame_metadata[key] = value
                metadata_dict_list.append(this_frame_metadata)

        # close H5 file
        f.close()

        # reshape if multiple images
        if (len(images.shape) == 3):  # type: ignore  # pragma: nocover
            # force reshape to 4 dimensions
            images = images.reshape((image_height, image_width, image_channels, 1))  # type: ignore

        # flip data (since it's upside down with displaying bottom-up (imshow origin="bottom"))
        images = np.flip(images, axis=0)  # type: ignore
    except Exception as e:
        if (file_obj["quiet"] is False):
            print("Error reading image file: %s" % (str(e)))
        problematic = True
        error_message = "error reading image file: %s" % (str(e))

    # return
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype
