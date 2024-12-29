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
import warnings
import random
import string
import cv2
import h5py
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

# static globals
__RGB_PGM_EXPECTED_HEIGHT = 480
__RGB_PGM_EXPECTED_WIDTH = 553
__RGB_PGM_DT = np.dtype("uint16")
__RGB_PGM_DT = __RGB_PGM_DT.newbyteorder('>')  # force big endian byte ordering
__RGB_PNG_DT = np.dtype("uint8")
__RGB_H5_DT = np.dtype("uint8")
__PNG_METADATA_PROJECT_UID = "trex"


def read(file_list, n_parallel=1, first_record=False, no_metadata=False, start_time=None, end_time=None, tar_tempdir=None, quiet=False):
    # throw up warning if the no_metadata flag is true and start/end times supplied
    #
    # NOTE: since the timestamp info is in the metadata, we can't filter based on
    # start or end time if we were told to not read the metadata, so we throw up
    # a warning and read all the data, and don't read the metadata (no_metadata being
    # true takes priority).
    if (no_metadata is True and (start_time is not None or end_time is not None)):
        warnings.warn("Cannot filter on start or end time if the no_metadata parameter is set to True. Set no_metadata to False " +
                      "to allow filtering on times. Will proceed by skipping filtering on times and returning no metadata.",
                      UserWarning,
                      stacklevel=1)
        start_time = None
        end_time = None

    # set tar path
    if (tar_tempdir is None):
        tar_tempdir = Path("%s/.trex_imager_readfile" % (str(Path.home())))
    os.makedirs(tar_tempdir, exist_ok=True)

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
        except ValueError:  # pragma: nocover
            # likely the read call is being used within a context that doesn't support the usage
            # of signals in this way, proceed without it
            pool = Pool(processes=n_parallel)

        # call readfile function, run each iteration with a single input file from file_list
        # NOTE: structure of data - data[file][metadata dictionary lists = 1, images = 0][frame]
        pool_data = []
        try:
            pool_data = pool.map(__trex_readfile_worker, processing_list)
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
            pool_data.append(__trex_readfile_worker(p))

    # set sizes
    image_width = pool_data[0][5]
    image_height = pool_data[0][6]
    image_channels = pool_data[0][7]
    image_dtype = pool_data[0][8]

    # derive number of frames to prepare for
    total_num_frames = 0
    for i in range(0, len(pool_data)):
        if (pool_data[i][2] is True):
            continue
        if (len(pool_data[i][0]) != 0):  # type: ignore
            if (image_channels > 1):
                total_num_frames += pool_data[i][0].shape[3]  # type: ignore
            else:
                total_num_frames += pool_data[i][0].shape[2]  # type: ignore

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
        if (image_channels > 1):
            this_num_frames = pool_data[i][0].shape[3]  # type: ignore
        else:
            this_num_frames = pool_data[i][0].shape[2]  # type: ignore

        # set task
        tasks.append((list_position, list_position + this_num_frames, i))

        # advance list position
        list_position = list_position + this_num_frames

    # pre-allocate array sizes
    if (image_channels > 1):
        images = np.empty([image_height, image_width, image_channels, list_position], dtype=image_dtype)
    else:
        images = np.empty([image_height, image_width, list_position], dtype=image_dtype)
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
        if (image_channels > 1):
            images[:, :, :, slice_idx1:slice_idx2] = pool_data[data_idx][0]  # type: ignore
        else:
            images[:, :, slice_idx1:slice_idx2] = pool_data[data_idx][0]  # type: ignore

        return data_idx

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        for t in tasks:
            executor.submit(assemble_data, t[0], t[1], t[2])

    # ensure entire array views as the desired dtype
    if (image_dtype == np.uint8):
        images = images.astype(np.uint8)
    elif (image_dtype == np.uint16):
        images = images.astype(np.uint16)

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
    image_dtype = np.dtype("uint8")  # type: ignore

    # check file extension to know how to process
    try:
        if (file_obj["filename"].endswith("pgm") or file_obj["filename"].endswith("pgm.gz")):
            return __rgb_readfile_worker_pgm(file_obj)
        elif (file_obj["filename"].endswith("png") or file_obj["filename"].endswith("png.tar")):
            return __rgb_readfile_worker_png(file_obj)
        elif (file_obj["filename"].endswith("h5")):
            return __rgb_readfile_worker_h5(file_obj)
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


def __rgb_readfile_worker_h5(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = __RGB_H5_DT

    # set start and end times so we can use shorter variable names lower down in this function
    start_time = file_obj["start_time"]
    end_time = file_obj["end_time"]

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
        if (len(idxs) == 0):
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
        if (len(images.shape) == 3):  # type: ignore
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


def __rgb_readfile_worker_png(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    problematic = False
    is_first = True
    error_message = ""
    image_width = 0
    image_height = 0
    image_channels = 0
    image_dtype = __RGB_PNG_DT
    working_dir_created = False

    # set up working dir
    this_working_dir = "%s/%s" % (file_obj["tar_tempdir"], ''.join(random.choices(string.ascii_lowercase, k=8)))  # nosec

    # set start and end times so we can use shorter variable names lower down in this function
    start_time = file_obj["start_time"]
    end_time = file_obj["end_time"]

    # extract start and end times of the filename
    #
    # NOTE: we use this to better inform us about zero-filesize files
    try:
        file_dt = datetime.datetime.strptime(os.path.basename(file_obj["filename"])[0:13], "%Y%m%d_%H%M")
    except Exception:
        if (file_obj["quiet"] is False):
            print("Failed to extract timestamp from filename")
        problematic = True
        error_message = "failed to extract timestamp from filename"
        return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
            image_width, image_height, image_channels, image_dtype

    # check if it's a tar file
    file_list = []
    if (file_obj["filename"].endswith(".png.tar")):
        # tar file, extract all frames and add to list
        tf = None
        try:
            tf = tarfile.open(file_obj["filename"])
            file_list = sorted(tf.getnames())
            if (file_obj["first_record"] is True):
                file_list = [file_list[0]]
                tf.extract(file_list[0], path=this_working_dir)  # nosec
            else:
                tf.extractall(path=this_working_dir)  # nosec
            for i in range(0, len(file_list)):
                file_list[i] = "%s/%s" % (this_working_dir, file_list[i])
            tf.close()
            working_dir_created = True
        except Exception as e:
            # cleanup
            try:
                shutil.rmtree(this_working_dir)
            except Exception:
                pass

            # set error message
            if (file_obj["quiet"] is False):
                print("Failed to open file '%s' " % (file_obj["filename"]))
            problematic = True
            error_message = "failed to open file: %s" % (str(e))
            try:
                if (tf is not None):
                    tf.close()
            except Exception:  # pragma: nocover
                pass
            return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
                image_width, image_height, image_channels, image_dtype
    else:
        # regular png
        file_list = [file_obj["filename"]]

    # read each png file
    for f in file_list:
        if (file_obj["no_metadata"] is False):
            # process metadata
            try:
                # set metadata values
                file_split = os.path.basename(f).split('_')
                site_uid = file_split[3]
                device_uid = file_split[4]
                exposure = "%.03f ms" % (float(file_split[5][:-2]))
                mode_uid = file_split[6][:-4]

                # set timestamp
                if ("burst" in f or "mode-b"):
                    timestamp = datetime.datetime.strptime("%sT%s.%s" % (file_split[0], file_split[1], file_split[2]), "%Y%m%dT%H%M%S.%f")
                else:
                    timestamp = datetime.datetime.strptime("%sT%s" % (file_split[0], file_split[1]), "%Y%m%dT%H%M%S")

                # check if we want to read this frame based on the start and end times
                if ((start_time is None or timestamp.replace(microsecond=0) >= start_time)
                        and (end_time is None or timestamp.replace(microsecond=0) <= end_time)):
                    pass
                else:
                    # don't want to read this file, continue on to the next
                    continue

                # set the metadata dict
                metadata_dict = {
                    "project_unique_id": __PNG_METADATA_PROJECT_UID,
                    "site_unique_id": site_uid,
                    "imager_unique_id": device_uid,
                    "mode_unique_id": mode_uid,
                    "image_request_start_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
                    "subframe_requested_exposure": exposure,
                }
                metadata_dict_list.append(metadata_dict)
            except Exception as e:
                if (file_obj["quiet"] is False):
                    print("Failed to read metadata from file '%s' " % (f))
                problematic = True
                error_message = "failed to read metadata: %s" % (str(e))
                break

        # read png file
        try:
            # read file
            image_np = cv2.imread(f, cv2.IMREAD_COLOR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            image_height = image_np.shape[0]
            image_width = image_np.shape[1]
            image_channels = image_np.shape[2] if len(image_np.shape) > 2 else 1
            if (image_channels > 1):
                image_matrix = np.reshape(image_np, (image_height, image_width, image_channels, 1))
            else:
                image_matrix = np.reshape(image_np, (image_height, image_width, 1))

            # initialize image stack
            if (is_first is True):
                images = image_matrix
                is_first = False
            else:
                if (image_channels > 1):
                    images = np.concatenate([images, image_matrix], axis=3)  # concatenate (on last axis)
                else:
                    images = np.dstack([images, image_matrix])  # depth stack images (on last axis)
        except Exception as e:
            if (file_obj["quiet"] is False):
                print("Failed reading image data frame: %s" % (str(e)))
            metadata_dict_list.pop()  # remove corresponding metadata entry
            problematic = True
            error_message = "image data read failure: %s" % (str(e))
            continue  # skip to next frame

    # cleanup
    #
    # NOTE: we only clean up the working dir if we created it
    if (working_dir_created is True):
        shutil.rmtree(this_working_dir)

    # check to see if the image is empty
    image_size_is_zero = False
    if (start_time is None and end_time is None):
        if (images.size == 0):
            image_size_is_zero = True
    elif (start_time is not None and end_time is not None):
        if (file_dt >= start_time and file_dt <= end_time):
            if (images.size == 0):
                image_size_is_zero = True
    elif (start_time is not None and file_dt >= start_time):
        if (images.size == 0):
            image_size_is_zero = True
    elif (end_time is not None and file_dt <= end_time):
        if (images.size == 0):
            image_size_is_zero = True

    # react to image data being empty
    if (image_size_is_zero is True):
        if (file_obj["quiet"] is False):
            print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"
    else:
        # flip data (since it's upside down with displaying bottom-up (imshow origin="bottom"))
        images = np.flip(images, axis=0)

    # return
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype


def __rgb_readfile_worker_pgm(file_obj):
    # init
    images = np.array([])
    metadata_dict_list = []
    is_first = True
    metadata_dict = {}
    site_uid = ""
    device_uid = ""
    problematic = False
    error_message = ""
    image_width = __RGB_PGM_EXPECTED_WIDTH
    image_height = __RGB_PGM_EXPECTED_HEIGHT
    image_channels = 1
    image_dtype = np.dtype("uint16")

    # Set metadata values
    file_split = os.path.basename(file_obj["filename"]).split('_')
    site_uid = file_split[3]
    device_uid = file_split[4]

    # set start and end times so we can use shorter variable names lower down in this function
    start_time = file_obj["start_time"]
    end_time = file_obj["end_time"]

    # extract start and end times of the filename
    #
    # NOTE: we use this to better inform us about zero-filesize files
    try:
        file_dt = datetime.datetime.strptime(os.path.basename(file_obj["filename"])[0:13], "%Y%m%d_%H%M")
    except Exception:
        if (file_obj["quiet"] is False):
            print("Failed to extract timestamp from filename")
        problematic = True
        error_message = "failed to extract timestamp from filename"
        return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
            image_width, image_height, image_channels, image_dtype

    # check file extension to see if it's gzipped or not
    unzipped = None
    try:
        if file_obj["filename"].endswith("pgm.gz"):
            unzipped = gzip.open(file_obj["filename"], mode='rb')
        elif file_obj["filename"].endswith("pgm"):
            unzipped = open(file_obj["filename"], mode='rb')
        else:
            if (file_obj["quiet"] is False):
                print("Unrecognized file type: %s" % (file_obj["filename"]))
            problematic = True
            error_message = "Unrecognized file type"
            try:
                if (unzipped is not None):
                    unzipped.close()
            except Exception:  # pragma: nocover
                pass
            return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
                image_width, image_height, image_channels, image_dtype
    except Exception as e:
        if (file_obj["quiet"] is False):
            print("Failed to open file '%s' " % (file_obj["filename"]))
        problematic = True
        error_message = "failed to open file: %s" % (str(e))
        try:
            if (unzipped is not None):
                unzipped.close()
        except Exception:  # pragma: nocover
            pass
        return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
            image_width, image_height, image_channels, image_dtype

    # read the file
    prev_line = None
    line = None
    skip_this_record = False
    while True:
        # break out depending on first_record param
        if (file_obj["first_record"] is True and is_first is False):
            break

        # read a line
        try:
            prev_line = line
            line = unzipped.readline()
        except Exception as e:
            if (file_obj["quiet"] is False):
                print("Error reading before image data in file '%s'" % (file_obj["filename"]))
            problematic = True
            metadata_dict_list = []
            images = np.array([])
            error_message = "error reading before image data: %s" % (str(e))
            try:
                unzipped.close()
            except Exception:  # pragma: nocover
                pass
            return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
                image_width, image_height, image_channels, image_dtype

        # break loop at end of file
        if (line == b''):
            break

        # magic number; this is not a metadata or image line, exclude
        if (line.startswith(b'P5\n')):  # type: ignore
            continue

        # process line
        if (line.startswith(b'#"')):  # type: ignore
            if (file_obj["no_metadata"] is False):
                # metadata lines start with #"<key>"
                try:
                    line_decoded = line.decode("ascii")  # type: ignore
                except Exception as e:
                    # skip metadata line if it can't be decoded, likely corrupt file
                    if (file_obj["quiet"] is False):
                        print("Error decoding metadata line: %s (line='%s', file='%s')" % (str(e), line, file_obj["filename"]))
                    problematic = True
                    error_message = "error decoding metadata line: %s" % (str(e))
                    continue

                # split the key and value out of the metadata line
                line_decoded_split = line_decoded.split('"')
                key = line_decoded_split[1]
                value = line_decoded_split[2].strip()

                # add entry to dictionary
                if (key in metadata_dict):
                    # key already exists, turn existing value into list and append new value
                    if (isinstance(metadata_dict[key], list)):
                        # is a list already
                        metadata_dict[key].append(value)
                    else:
                        metadata_dict[key] = [metadata_dict[key], value]
                else:
                    # normal metadata value
                    metadata_dict[key] = value

                # split dictionaries up per frame, exposure plus initial readout is
                # always the end of metadata for frame
                if (key.startswith("Effective image exposure")):
                    # check if we want to skip this frame based on the start and end times
                    if ("image_request_start_timestamp" in metadata_dict):
                        this_timestamp = datetime.datetime.strptime(metadata_dict["image_request_start_timestamp"],
                                                                    "%Y-%m-%d %H:%M:%S.%f UTC").replace(microsecond=0)
                    elif ("Image request start" in metadata_dict):
                        this_timestamp = datetime.datetime.strptime(metadata_dict["Image request start"],
                                                                    "%Y-%m-%d %H:%M:%S.%f UTC").replace(microsecond=0)
                    else:
                        if (file_obj["quiet"] is False):
                            print("Unexpected timestamp metadata format in %s" % (file_obj["filename"]))
                        problematic = True
                        error_message = "unexpected timestamp metadata format"
                        continue
                    if ((start_time is None or this_timestamp >= start_time) and (end_time is None or this_timestamp <= end_time)):
                        metadata_dict_list.append(metadata_dict)
                    else:
                        skip_this_record = True
                    metadata_dict = {}
        elif line == b'65535\n':
            # there are 2 lines between "exposure plus read out" and the image
            # data, the first is the image dimensions and the second is the max
            # value
            #
            # check the previous line to get the dimensions of the image
            prev_line_split = prev_line.decode("ascii").strip().split()  # type: ignore
            image_width = int(prev_line_split[0])
            image_height = int(prev_line_split[1])
            bytes_to_read = image_width * image_height * 2  # 16-bit image depth

            # read image
            try:
                # skip the image data if we wanted to
                if (skip_this_record is True):
                    skip_this_record = False
                    unzipped.seek(bytes_to_read, 1)
                    continue

                # read the image size in bytes from the file
                image_bytes = unzipped.read(bytes_to_read)

                # format bytes into numpy array of unsigned shorts (2byte numbers, 0-65536),
                # effectively an array of pixel values
                #
                # NOTE: this is set to a different dtype that what we return on purpose.
                image_np = np.frombuffer(image_bytes, dtype=__RGB_PGM_DT)  # type: ignore

                # change 1d numpy array into matrix with correctly located pixels
                image_matrix = np.reshape(image_np, (image_height, image_width, 1))
            except Exception as e:
                if (file_obj["quiet"] is False):
                    print("Failed reading image data frame: %s" % (str(e)))
                metadata_dict_list.pop()  # remove corresponding metadata entry
                problematic = True
                error_message = "image data read failure: %s" % (str(e))
                continue  # skip to next frame

            # initialize image stack
            if (is_first is True):
                images = image_matrix
                is_first = False
            else:
                images = np.dstack([images, image_matrix])  # depth stack images (on 3rd axis)

    # close gzip file
    unzipped.close()

    # set the site/device uids, or inject the site and device UIDs if they are missing
    if ("Site unique ID" not in metadata_dict):
        metadata_dict["Site unique ID"] = site_uid
    if ("Imager unique ID" not in metadata_dict):
        metadata_dict["Imager unique ID"] = device_uid

    # check to see if the image is empty
    image_size_is_zero = False
    if (start_time is None and end_time is None):
        if (images.size == 0):
            image_size_is_zero = True
    elif (start_time is not None and end_time is not None):
        if (file_dt >= start_time and file_dt <= end_time):
            if (images.size == 0):
                image_size_is_zero = True
    elif (start_time is not None and file_dt >= start_time):
        if (images.size == 0):
            image_size_is_zero = True
    elif (end_time is not None and file_dt <= end_time):
        if (images.size == 0):
            image_size_is_zero = True

    # react to image data being empty
    if (image_size_is_zero is True):
        if (file_obj["quiet"] is False):
            print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"
    else:
        # flip data (since it's upside down with displaying bottom-up (imshow origin="bottom"))
        images = np.flip(images, axis=0)

    # return
    return images, metadata_dict_list, problematic, file_obj["filename"], error_message, \
        image_width, image_height, image_channels, image_dtype
