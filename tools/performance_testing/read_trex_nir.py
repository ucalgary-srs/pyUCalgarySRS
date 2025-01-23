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

import pyucalgarysrs
import datetime
import argparse
import numpy as np
from tqdm import tqdm


def main():
    # args
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of iterations")
    parser.add_argument("--nparallel", type=int, default=2, help="Number of parallel workers for reading")
    args = parser.parse_args()

    # init
    print("[%s] Initializing" % (datetime.datetime.now()))
    srs = pyucalgarysrs.PyUCalgarySRS()
    start_dt = datetime.datetime(2021, 11, 4, 6, 0, 0)
    end_dt = datetime.datetime(2021, 11, 4, 6, 59, 59)
    print("[%s] Downloading data" % (datetime.datetime.now()))
    res = srs.data.download("TREX_NIR_RAW", start_dt, end_dt, site_uid="gill", progress_bar_disable=True)

    # read data
    times = []
    for _ in tqdm(range(0, args.count), desc="[%s] Reading data  " % (datetime.datetime.now()), ncols=125, ascii=" #"):
        d1 = datetime.datetime.now()
        srs.data.read(res.dataset, res.filenames, n_parallel=args.nparallel)
        times.append((datetime.datetime.now() - d1).total_seconds())
    times = np.asarray(times)

    # print
    print("[%s] Completed in %.03fs, avg %.03f s += %.03f ms" % (
        datetime.datetime.now(),
        np.sum(times),
        np.mean(times),
        np.std(times) * 1000.0,
    ))


if (__name__ == "__main__"):
    main()
