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
import signal
import warnings
from typing import List
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from scipy.io import readsav
from ...exceptions import SRSError


def read(file_list, n_parallel=1, quiet=False) -> List:
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
        data = []
        try:
            data = pool.map(partial(__skymap_readfile_worker, quiet=quiet), file_list)
        except KeyboardInterrupt:  # pragma: nocover
            pool.terminate()  # gracefully kill children
            return []
        else:
            pool.close()
            pool.join()
    else:
        # don't bother using multiprocessing with one worker, just call the worker function directly
        data = []
        for f in file_list:
            data.append(__skymap_readfile_worker(f, quiet=quiet))

    # process pool results
    data_dict_list = []
    for i in range(0, len(data)):
        if (data[i][0] is not None):
            data[i][0]["filename"] = data[i][2]
            data_dict_list.append(data[i][0])
        if (data[i][1] is True):
            raise SRSError("Error reading skymap file '%s'" % (os.path.basename(data[i][2])))
    data = None

    # return
    return data_dict_list


def __skymap_readfile_worker(file, quiet=False):
    # init
    data_recarray = {}

    # convert to str to handle path type
    file = str(file)

    # try to read in the file
    try:
        # NOTE: we suppress some warnings that the readsav routine
        # outputs since they only apply to some of our THEMIS
        # skymap (circa 2007-2012) and do not affect the data
        # which is loaded.
        warnings.simplefilter("ignore", category=Warning)

        # read the save file
        data_recarray = readsav(file, python_dict=True)
    except Exception as e:
        if (quiet is False):
            print("Failed to read file '%s'" % (file))
        problematic = True
        error_message = "Failed to read file: %s" % (str(e))
        return {}, problematic, file, error_message

    # return
    #
    # NOTE: order is --> data, problematic flag, filename, error message
    return data_recarray, False, file, None
