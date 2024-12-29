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

import gzip
import numpy as np
import signal
import os
import warnings
import datetime
import h5py
from multiprocessing import Pool
from functools import partial
from concurrent.futures import ThreadPoolExecutor

# globals
__SPECTROGRAPH_EXPECTED_HEIGHT = 1024
__SPECTROGRAPH_EXPECTED_WIDTH = 256
__SPECTROGRAPH_RAW_DT = np.dtype("uint16")
__SPECTROGRAPH_RAW_DT = __SPECTROGRAPH_RAW_DT.newbyteorder('>')  # force big endian byte ordering
__SPECTROGRAPH_PROCESSED_DT = np.dtype("float32")


def read_raw(file_list, n_parallel=1, first_record=False, no_metadata=False, start_time=None, end_time=None, quiet=False):
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
            data = pool.map(
                partial(
                    __spectrograph_raw_readfile_worker,
                    first_record=first_record,
                    no_metadata=no_metadata,
                    start_time=start_time,
                    end_time=end_time,
                    quiet=quiet,
                ), file_list)
        except KeyboardInterrupt:  # pragma: nocover
            pool.terminate()  # gracefully kill children
            return np.empty((0, 0, 0), dtype=__SPECTROGRAPH_RAW_DT), [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        data = []
        for f in file_list:
            data.append(
                __spectrograph_raw_readfile_worker(
                    f,
                    first_record=first_record,
                    no_metadata=no_metadata,
                    start_time=start_time,
                    end_time=end_time,
                    quiet=quiet,
                ))

    # derive number of frames to prepare for
    total_num_frames = 0
    image_width = __SPECTROGRAPH_EXPECTED_WIDTH
    image_height = __SPECTROGRAPH_EXPECTED_HEIGHT
    for i in range(0, len(data)):
        if (data[i][2] is True):
            continue
        if (len(data[i][0]) != 0):
            total_num_frames += data[i][0].shape[2]
            image_height = data[i][0].shape[0]
            image_width = data[i][0].shape[1]

    # set tasks
    list_position = 0
    tasks = []
    problematic_file_list = []
    for i in range(0, len(data)):
        # check if file was problematic
        if (data[i][2] is True):
            problematic_file_list.append({
                "filename": data[i][3],
                "error_message": data[i][4],
            })
            continue

        # check if any data was read in
        if (len(data[i][0]) == 0):
            continue

        # find actual number of frames, this may differ from predicted due to dropped frames, end
        # or start of imaging
        real_num_frames = data[i][0].shape[2]

        # set task
        tasks.append((list_position, list_position + real_num_frames, i))

        # advance list position
        list_position = list_position + real_num_frames

    # pre-allocate array sizes
    images = np.empty([image_height, image_width, list_position], dtype=__SPECTROGRAPH_RAW_DT)
    if (no_metadata is False):
        metadata_dict_list = [{}] * list_position
    else:
        metadata_dict_list = []

    # merge data using number of threads
    def assemble_data(slice_idx1, slice_idx2, data_idx):
        # merge metadata
        if (no_metadata is False):
            metadata_dict_list[slice_idx1:slice_idx2] = data[data_idx][1]  # type: ignore

        # merge image data
        images[:, :, slice_idx1:slice_idx2] = data[data_idx][0]  # type: ignore

        return data_idx

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        for t in tasks:
            executor.submit(assemble_data, t[0], t[1], t[2])

    # ensure entire array views as uint16
    images = images.astype(np.uint16)

    # return
    data = None
    return images, metadata_dict_list, problematic_file_list


def __spectrograph_raw_readfile_worker(file, first_record=False, no_metadata=False, start_time=None, end_time=None, quiet=False):
    # init
    images = np.array([])
    metadata_dict_list = []
    is_first = True
    metadata_dict = {}
    site_uid = ""
    device_uid = ""
    problematic = False
    error_message = ""

    # convert to str to handle path type
    file = str(file)

    # extract start and end times of the filename
    #
    # NOTE: we use this to better inform us about zero-filesize files
    try:
        file_dt = datetime.datetime.strptime(os.path.basename(file)[0:13], "%Y%m%d_%H%M")
    except Exception:
        if (quiet is False):
            print("Failed to extract timestamp from filename")
        problematic = True
        error_message = "failed to extract timestamp from filename"
        return images, metadata_dict_list, problematic, file, error_message

    # set site UID and device UID in case we need it (ie. dark frames, or unstacked files)
    file_split = os.path.basename(file).split('_')
    if (len(file_split) == 5):
        # is a regular file
        site_uid = file_split[2]
        device_uid = file_split[3]
    elif (len(file_split) > 5):
        # is likely a dark frame or a unstacked frame
        site_uid = file_split[3]
        device_uid = file_split[4]

    # check file extension to see if it's gzipped or not
    unzipped = None
    try:
        if file.endswith("pgm.gz"):
            unzipped = gzip.open(file, mode='rb')
        elif file.endswith("pgm"):
            unzipped = open(file, mode='rb')
        else:
            if (quiet is False):
                print("Unrecognized file type: %s" % (file))
            problematic = True
            error_message = "Unrecognized file type"
            try:
                if (unzipped is not None):
                    unzipped.close()
            except Exception:  # pragma: nocover
                pass
            return images, metadata_dict_list, problematic, file, error_message
    except Exception as e:
        if (quiet is False):
            print("Failed to open file '%s' " % (file))
        problematic = True
        error_message = "failed to open file: %s" % (str(e))
        try:
            if (unzipped is not None):
                unzipped.close()
        except Exception:  # pragma: nocover
            pass
        return images, metadata_dict_list, problematic, file, error_message

    # read the file
    prev_line = None
    line = None
    skip_this_record = False
    while True:
        # break out depending on first_record param
        if (first_record is True and is_first is False):
            break

        # read a line
        try:
            prev_line = line
            line = unzipped.readline()
        except Exception as e:
            if (quiet is False):
                print("Error reading before image data in file '%s'" % (file))
            problematic = True
            metadata_dict_list = []
            images = np.array([])
            error_message = "error reading before image data: %s" % (str(e))
            try:
                unzipped.close()
            except Exception:  # pragma: nocover
                pass
            return images, metadata_dict_list, problematic, file, error_message

        # break loop at end of file
        if (line == b''):
            break

        # magic number; this is not a metadata or image line, exclude
        if (line.startswith(b'P5\n')):  # type: ignore
            continue

        # process line
        if (line.startswith(b'#"')):  # type: ignore
            if (no_metadata is False):
                # metadata lines start with #"<key>"
                try:
                    line_decoded = line.decode("ascii")  # type: ignore
                except Exception as e:
                    # skip metadata line if it can't be decoded, likely corrupt file
                    if (quiet is False):
                        print("Error decoding metadata line: %s (line='%s', file='%s')" % (str(e), line, file))
                    problematic = True
                    error_message = "error decoding metadata line: %s" % (str(e))
                    continue

                # split the key and value out of the metadata line
                line_decoded_split = line_decoded.split('"')
                key = line_decoded_split[1]
                value = line_decoded_split[2].strip()

                # add entry to dictionary
                metadata_dict[key] = value

                # set the site/device uids, or inject the site and device UIDs if they are missing
                if ("Site unique ID" not in metadata_dict):
                    metadata_dict["Site unique ID"] = site_uid
                else:
                    site_uid = metadata_dict["Site unique ID"]
                if ("Imager unique ID" not in metadata_dict):
                    metadata_dict["Imager unique ID"] = device_uid
                else:
                    device_uid = metadata_dict["Imager unique ID"]

                # split dictionaries up per frame, exposure plus initial readout is
                # always the end of metadata for frame
                if (key.startswith("Exposure plus readout")):
                    # check if we want to skip this frame based on the start and end times
                    this_timestamp = datetime.datetime.strptime(metadata_dict["Image request start"],
                                                                "%Y-%m-%d %H:%M:%S.%f UTC").replace(microsecond=0)
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
                image_np = np.frombuffer(image_bytes, dtype=__SPECTROGRAPH_RAW_DT)  # type: ignore

                # change 1d numpy array into matrix with correctly located pixels
                image_matrix = np.reshape(image_np, (image_height, image_width, 1))
            except Exception as e:
                if (quiet is False):
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
    if (image_size_is_zero is True):
        if (quiet is False):
            print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"

    # return
    return images, metadata_dict_list, problematic, file, error_message


def __str_to_datetime_formatter(timestamp_str):
    return datetime.datetime.strptime(timestamp_str.decode(), "%Y-%m-%d %H:%M:%S UTC")


def read_processed(file_list, n_parallel=1, first_record=False, no_metadata=False, start_time=None, end_time=None, quiet=False):
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
            data = pool.map(
                partial(
                    __spectrograph_processed_readfile_worker,
                    first_record=first_record,
                    no_metadata=no_metadata,
                    start_time=start_time,
                    end_time=end_time,
                    quiet=quiet,
                ), file_list)
        except KeyboardInterrupt:  # pragma: nocover
            pool.terminate()  # gracefully kill children
            return np.empty((0, 0, 0), dtype=__SPECTROGRAPH_PROCESSED_DT), np.empty((0), dtype=datetime.datetime), [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        data = []
        for f in file_list:
            data.append(
                __spectrograph_processed_readfile_worker(
                    f,
                    first_record=first_record,
                    no_metadata=no_metadata,
                    start_time=start_time,
                    end_time=end_time,
                    quiet=quiet,
                ))

    # derive number of frames to prepare for
    total_num_frames = 0
    image_width = __SPECTROGRAPH_EXPECTED_WIDTH
    image_height = __SPECTROGRAPH_EXPECTED_HEIGHT
    for i in range(0, len(data)):
        if (data[i][3] is True):
            continue
        if (len(data[i][0]) != 0):  # type: ignore
            total_num_frames += data[i][0].shape[2]  # type: ignore
            image_height = data[i][0].shape[0]  # type: ignore
            image_width = data[i][0].shape[1]  # type: ignore

    # pre-allocate array sizes
    spectra = np.empty((image_height, image_width, total_num_frames), dtype=__SPECTROGRAPH_PROCESSED_DT)
    timestamp = np.empty((total_num_frames), dtype=object)
    if (no_metadata is False):
        metadata_dict_list = [{}] * total_num_frames
    else:
        metadata_dict_list = []
    problematic_file_list = []

    # populate data
    list_position = 0
    for i in range(0, len(data)):
        # check if file was problematic
        if (data[i][3] is True):
            problematic_file_list.append({
                "filename": data[i][4],
                "error_message": data[i][5],
            })
            continue

        # check if any data was read in
        if (len(data[i][0]) == 0):  # type: ignore
            continue

        # find actual number of frames, this may differ from predicted due to
        # dropped frames, end or start of imaging
        real_num_frames = data[i][0].shape[2]  # type: ignore

        # populate image data
        spectra[:, :, list_position:list_position + real_num_frames] = data[i][0]

        # populate timestamps
        timestamp[list_position:list_position + real_num_frames] = data[i][1]

        # populate metadata objects
        if (no_metadata is False):
            metadata_dict_list[list_position:list_position + real_num_frames] = data[i][2]

        # advance list position
        list_position = list_position + real_num_frames

    # trim unused elements from predicted array sizes
    if (no_metadata is False):
        metadata_dict_list = metadata_dict_list[0:list_position]
    spectra = np.delete(spectra, range(list_position, total_num_frames), axis=2)

    # ensure entire array views as float32
    spectra = spectra.astype(__SPECTROGRAPH_PROCESSED_DT)

    # convert timestamps to datetime objections
    timestamp = np.vectorize(__str_to_datetime_formatter)(timestamp)  # type: ignore
    timestamp = timestamp.astype(datetime.datetime)

    # return
    data = None
    return spectra, timestamp, metadata_dict_list, problematic_file_list


def __spectrograph_processed_readfile_worker(file, first_record=False, no_metadata=False, start_time=None, end_time=None, quiet=False):
    # init
    spectra = np.array([])
    timestamps = np.array([])
    metadata_dict_list = []
    problematic = False
    error_message = ""

    # process file
    try:
        # open H5 file
        f = h5py.File(file, 'r')

        # get timestamps
        if (first_record is True):
            original_timestamps = [f["data"]["timestamp"][0]]  # type: ignore
        else:
            original_timestamps = f["data"]["timestamp"][:]  # type: ignore

        # search through timestamps for indexes to read data for, based
        # on the start and end times
        idxs = []
        timestamps = []
        if (start_time is None and end_time is None):
            # no filtering, read all indexes
            idxs = np.arange(0, len(original_timestamps))  # type: ignore
            timestamps = original_timestamps
        else:
            for i, val in enumerate(original_timestamps):  # type: ignore
                val_dt = datetime.datetime.strptime(val.decode(), "%Y-%m-%d %H:%M:%S UTC")
                if ((start_time is None or val_dt >= start_time) and (end_time is None or val_dt <= end_time)):
                    # matches start and end time range, we want to read this index
                    idxs.append(i)
                    timestamps.append(original_timestamps[i])  # type: ignore

        # bail out if we don't want to read any frames
        if (len(idxs) == 0):
            timestamps = np.array([])
            return spectra, timestamps, metadata_dict_list, problematic, file, error_message

        # get spectra
        spectra = f["data"]["spectra"][:, :, idxs]  # type: ignore

        # read metadata
        file_metadata = {}
        if (no_metadata is False):
            # get file metadata
            for key, value in f["metadata"]["file"].attrs.items():  # type: ignore
                file_metadata[key] = value

            # read wavelength
            file_metadata["wavelength"] = f["data"]["wavelength"][:]  # type: ignore
            metadata_dict_list = [file_metadata] * len(idxs)  # type: ignore

        # close H5 file
        f.close()

        # set image vars and reshape if multiple images
        image_height = spectra.shape[0]  # type: ignore
        image_width = spectra.shape[1]  # type: ignore
        if (len(spectra.shape) == 2):  # type: ignore
            # force reshape to 3 dimensions
            spectra = spectra.reshape((image_height, image_width, 1))  # type: ignore
    except Exception as e:
        if (quiet is False):
            print("Error reading image file: %s" % (str(e)))
        problematic = True
        error_message = "error reading image file: %s" % (str(e))

    # return
    return spectra, timestamps, metadata_dict_list, problematic, file, error_message
