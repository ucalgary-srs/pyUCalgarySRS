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
import gzip
import shutil
import signal
import tarfile
import random
import string
import cv2
import h5py
import numpy as np
from pathlib import Path
from multiprocessing import Pool

# static globals
__RGB_PGM_EXPECTED_HEIGHT = 480 
__RGB_PGM_EXPECTED_WIDTH = 553
__RGB_PGM_DT = np.dtype("uint16")
__RGB_PGM_DT = __RGB_PGM_DT.newbyteorder('>')  # force big endian byte ordering
__RGB_PNG_DT = np.dtype("uint8")
__RGB_H5_DT = np.dtype("uint8")
__PNG_METADATA_PROJECT_UID = "trex"


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

    # pre-allocate array sizes
    if (image_channels > 1):
        images = np.empty([image_height, image_width, image_channels, total_num_frames], dtype=image_dtype)
    else:
        images = np.empty([image_height, image_width, total_num_frames], dtype=image_dtype)
    metadata_dict_list = [{}] * total_num_frames
    problematic_file_list = []

    # populate data
    list_position = 0
    for i in range(0, len(pool_data)):
        # check if file was problematic
        if (pool_data[i][2] is True):
            problematic_file_list.append({
                "filename": pool_data[i][3],
                "error_message": pool_data[i][4],
            })
            continue

        # check if any data was read in
        if (len(pool_data[i][1]) == 0):
            continue

        # find actual number of frames, this may differ from predicted due to dropped frames, end
        # or start of imaging
        if (image_channels > 1):
            this_num_frames = pool_data[i][0].shape[3]  # type: ignore
        else:
            this_num_frames = pool_data[i][0].shape[2]  # type: ignore

        # metadata dictionary list at data[][1]
        metadata_dict_list[list_position:list_position + this_num_frames] = pool_data[i][1]
        if (image_channels > 1):
            images[:, :, :, list_position:list_position + this_num_frames] = pool_data[i][0]
        else:
            images[:, :, list_position:list_position + this_num_frames] = pool_data[i][0]
        list_position = list_position + this_num_frames  # advance list position

    # trim unused elements from predicted array sizes
    metadata_dict_list = metadata_dict_list[0:list_position]
    if (image_channels > 1):
        images = np.delete(images, range(list_position, total_num_frames), axis=3)
    else:
        images = np.delete(images, range(list_position, total_num_frames), axis=2)

    # ensure entire array views as the desired dtype
    if (image_dtype == np.uint8):
        images = images.astype(np.uint8)
    elif (image_dtype == np.uint16):
        images = images.astype(np.uint16)

    # return
    pool_data = None
    return images, metadata_dict_list, problematic_file_list


def __grid_readfile_worker(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = np.dtype("uint8")  # type: ignore

    # check file extension to know how to process
    try:
        if (file_obj["filename"].endswith("h5")):
            return __grid_readfile_worker_h5(file_obj)
        else:
            if (file_obj["quiet"] is False):
                print("Unrecognized file type: %s" % (file_obj["filename"]))
            problematic = True
            error_message = "Unrecognized file type"
    except Exception as e:
        if (file_obj["quiet"] is False):
            print("Failed to process file '%s' " % (file_obj["filename"]))
        problematic = True
        error_message = "failed to process file: %s" % (str(e))
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype


def __grid_readfile_worker_h5(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = __RGB_H5_DT

    # open H5 file
    f = h5py.File(file_obj["filename"], 'r')

    # get images and timestamps
    if (file_obj["first_record"] is True):
        # get only first frame
        images = f["data"]["images"][:, :, :, 0]  # type: ignore
        timestamps = [f["data"]["timestamp"][0]]  # type: ignore
    else:
        # get all frames
        images = f["data"]["images"][:]  # type: ignore
        timestamps = f["data"]["timestamp"][:]  # type: ignore

    # read metadata
    file_metadata = {}
    if (file_obj["no_metadata"] is True):
        metadata_dict_list = [{}] * len(timestamps)  # type: ignore
    else:
        # get file metadata
        for key, value in f["metadata"]["file"].attrs.items():  # type: ignore
            file_metadata[key] = value

        # read frame metadata
        for i in range(0, len(timestamps)):  # type: ignore
            this_frame_metadata = file_metadata.copy()
            for key, value in f["metadata"]["frame"]["frame%d" % (i)].attrs.items():  # type: ignore
                this_frame_metadata[key] = value
            metadata_dict_list.append(this_frame_metadata)

    # close H5 file
    f.close()

    # set image vars and reshape if multiple images
    image_height = images.shape[0]  # type: ignore
    image_width = images.shape[1]  # type: ignore
    image_channels = images.shape[2]  # type: ignore
    if (len(images.shape) == 3):  # type: ignore
        # force reshape to 4 dimensions
        images = images.reshape((image_height, image_width, image_channels, 1))  # type: ignore

    # flip data (since it's upside down with displaying bottom-up (imshow origin="bottom"))
    images = np.flip(images, axis=0)  # type: ignore

    # return
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype
