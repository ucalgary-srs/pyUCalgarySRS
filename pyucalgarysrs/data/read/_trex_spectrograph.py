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
from pathlib import Path
from multiprocessing import Pool
from functools import partial

# globals
__SPECTROGRAPH_EXPECTED_HEIGHT = 1024
__SPECTROGRAPH_EXPECTED_WIDTH = 256
__SPECTROGRAPH_DT = np.dtype("uint16")
__SPECTROGRAPH_DT = __SPECTROGRAPH_DT.newbyteorder('>')  # force big endian byte ordering


def read(file_list, n_parallel=1, first_record=False, no_metadata=False, quiet=False):
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
                __spectrograph_readfile_worker,
                first_record=first_record,
                no_metadata=no_metadata,
                quiet=quiet,
            ), file_list)
        except KeyboardInterrupt:  # pragma: nocover
            pool.terminate()  # gracefully kill children
            return np.empty((0, 0, 0), dtype=__SPECTROGRAPH_DT), [], []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        data = []
        for f in file_list:
            data.append(__spectrograph_readfile_worker(
                f,
                first_record=first_record,
                no_metadata=no_metadata,
                quiet=quiet,
            ))

    # derive number of frames to prepare for
    total_num_frames = 0
    image_width = __SPECTROGRAPH_EXPECTED_WIDTH
    image_height = __SPECTROGRAPH_EXPECTED_HEIGHT
    for i in range(0, len(data)):
        if (data[i][2] is True):
            continue
        total_num_frames += data[i][0].shape[2]
        image_height = data[i][0].shape[0]
        image_width = data[i][0].shape[1]

    # pre-allocate array sizes
    images = np.empty([image_height, image_width, total_num_frames], dtype=__SPECTROGRAPH_DT)
    metadata_dict_list = [{}] * total_num_frames
    problematic_file_list = []

    # populate data
    list_position = 0
    for i in range(0, len(data)):
        # check if file was problematic
        if (data[i][2] is True):
            problematic_file_list.append({
                "filename": data[i][3],
                "error_message": data[i][4],
            })
            continue

        # check if any data was read in
        if (len(data[i][1]) == 0):
            continue

        # find actual number of frames, this may differ from predicted due to dropped frames, end
        # or start of imaging
        real_num_frames = data[i][0].shape[2]

        # metadata dictionary list at data[][1]
        metadata_dict_list[list_position:list_position + real_num_frames] = data[i][1]
        images[:, :, list_position:list_position + real_num_frames] = data[i][0]  # image arrays at data[][0]
        list_position = list_position + real_num_frames  # advance list position

    # trim unused elements from predicted array sizes
    metadata_dict_list = metadata_dict_list[0:list_position]
    images = np.delete(images, range(list_position, total_num_frames), axis=2)

    # ensure entire array views as uint16
    images = images.astype(np.uint16)

    # return
    data = None
    return images, metadata_dict_list, problematic_file_list


def __spectrograph_readfile_worker(file, first_record=False, no_metadata=False, quiet=False):
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
            if (no_metadata is True):
                metadata_dict = {}
                metadata_dict_list.append(metadata_dict)
            else:
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
                    metadata_dict_list.append(metadata_dict)
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
                # read the image size in bytes from the file
                image_bytes = unzipped.read(bytes_to_read)

                # format bytes into numpy array of unsigned shorts (2byte numbers, 0-65536),
                # effectively an array of pixel values
                image_np = np.frombuffer(image_bytes, dtype=__SPECTROGRAPH_DT)  # type: ignore

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
    if (images.size == 0):
        if (quiet is False):
            print("Error reading image file: found no image data")
        problematic = True
        error_message = "no image data"

    # return
    return images, metadata_dict_list, problematic, file, error_message
