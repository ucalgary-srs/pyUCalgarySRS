#! /usr/bin/env python
#
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
"""
This script builds the test data that the test suite reads from disk, into
tests/test_data. It is what 'make get-test-data' runs.

Every file the test suite refers to falls into one of these categories:

  1) the archive manifest (IMAGER_FILES, GRID_FILES, and the lists below it) -- a
     file that still exists in the open data platform, and is downloaded verbatim.

  2) LEGACY_RGB_* and BURST_720P_FILE -- TREx RGB data in the legacy formats (a PGM
     of 16-bit frames, a tarball of PNGs, and the 720p 'mode-b3' burst mode). All
     TREx RGB raw data in the archive has since been reprocessed into HDF5, so these
     are regenerated in the legacy formats using the imagery from the HDF5 file for
     the same minute.

  3) WARNING_FILE -- a REGO file with a malformed metadata line, which the reader
     warns about but reads anyway. The archive's copy has since been fixed, so the
     malformed line is put back.

  4) DECOMPRESSED_FILES / TAR_MEMBER_FILES / COPIED_FILES -- files made from one of
     the above: decompressed copies, the single frames pulled out of a burst tarball,
     and the copies that the 'badperms' tests chmod to 000 and back.

  5) EMPTY_FILES / TRUNCATED_FILES / SHORT_FRAME_FILES -- the files that the test
     suite expects to fail reading. These were broken files that turned up in the
     archive over the years, most of which have since been removed from it. Each of
     the three lists recreates a different way of being broken, and between them they
     cover the three error paths the readers have for it: no image data at all, the
     compressed stream ending while metadata is being read, and the last image frame
     being short.

  6) PLACEHOLDER_FILE_DIRS -- the 'not a data file at all' file that each reader gets
     pointed at.

Usage:

  python3 tools/build_test_data.py [--clean]

Files that are already in place are left alone, so an interrupted build only fetches
what it missed. Intermediate downloads are cached in <data dir>/.build_cache. Pass
--tarball to also package the tree up, for hosting it somewhere or handing it to
someone.
"""

import argparse
import glob
import gzip
import os
import shutil
import subprocess  # nosec
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# globals
DATA_TREE = "https://data.phys.ucalgary.ca/sort_by_project"
TARBALL_FILENAME = "pyucalgarysrs_test_data.tar.gz"
DEFAULT_DATA_DIR = "%s/tests/test_data" % (os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
WORK_DIR_NAME = ".build_cache"
RGB_LEGACY_HEIGHT = 480
RGB_LEGACY_WIDTH = 553
RGB_LEGACY_FRAMES = 20
RGB_LEGACY_CADENCE_SECONDS = 3
SHORT_FRAME_MARKER = b"65535\n"
SHORT_FRAME_KEEP_BYTES = 1000

# files that are downloaded from the open data platform as-is, grouped by where they live in the data tree
# raw imager data: <data tree>/<path>/<yyyy>/<mm>/<dd>/<site>_<device>/ut<hh>/<filename>
IMAGER_FILES = {
    "read_rego": (
        "GO-Canada/REGO/stream0",
        [
            "20180403_0600_gill_rego-652_6300.pgm.gz",
            "20180403_0601_gill_rego-652_6300.pgm.gz",
            "20180403_0602_gill_rego-652_6300.pgm.gz",
            "20180403_0603_gill_rego-652_6300.pgm.gz",
            "20180403_0604_gill_rego-652_6300.pgm.gz",
            "20180403_0605_gill_rego-652_6300.pgm.gz",
        ],
    ),
    "read_smile": (
        "SMILE/asi/l0/raw",
        [
            "20250315_0600_atha_smile-31_rgb-full.h5",
            "20250315_0601_atha_smile-31_rgb-full.h5",
            "20250315_0602_atha_smile-31_rgb-full.h5",
            "20250315_0603_atha_smile-31_rgb-full.h5",
            "20250315_0604_atha_smile-31_rgb-full.h5",
        ],
    ),
    "read_smile/single_frame": (
        "SMILE/asi/l0/raw",
        [
            "20250921_020000_000053_pina_smile-05_raw.h5",
            "20250921_020003_000076_pina_smile-05_raw.h5",
            "20250921_020006_000053_pina_smile-05_raw.h5",
        ],
    ),
    "read_themis": (
        "THEMIS/asi/stream0",
        [
            "20040825_0526_atha_themis01_full_1000ms.pgm.bz2",
            "20140310_0600_gill_themis19_full.pgm.gz",
            "20140310_0601_gill_themis19_full.pgm.gz",
            "20140310_0602_gill_themis19_full.pgm.gz",
            "20140310_0603_gill_themis19_full.pgm.gz",
            "20140310_0604_gill_themis19_full.pgm.gz",
            "20140310_0605_gill_themis19_full.pgm.gz",
            "20230104_0606_talo_themis15_full.pgm.gz",
        ],
    ),
    "read_trex_blue": (
        "TREx/blueline/stream0",
        [
            "20220308_0600_gill_blue-814_full.pgm.gz",
            "20220308_0601_gill_blue-814_full.pgm.gz",
            "20220308_0602_gill_blue-814_full.pgm.gz",
            "20220308_0603_gill_blue-814_full.pgm.gz",
            "20220308_0604_gill_blue-814_full.pgm.gz",
            "20220308_0605_gill_blue-814_full.pgm.gz",
            "20240101_005115_825600_atha_blue-633_dark_full.pgm",
            "20240101_0600_atha_blue-633_full.pgm.gz",
        ],
    ),
    "read_trex_nir": (
        "TREx/NIR/stream0",
        [
            "20200101_0700_gill_nir-217_8446.pgm.gz",
            "20220101_2329_gill_nir-216_8446_dark.pgm.gz",
            "20220210_1043_atha_nir-221_8446.pgm.gz",
            "20220307_0600_gill_nir-216_8446.pgm.gz",
            "20220307_0601_gill_nir-216_8446.pgm.gz",
            "20220307_0602_gill_nir-216_8446.pgm.gz",
            "20220307_0603_gill_nir-216_8446.pgm.gz",
            "20220307_0604_gill_nir-216_8446.pgm.gz",
            "20220307_0605_gill_nir-216_8446.pgm.gz",
        ],
    ),
    "read_trex_rgb/stream0": (
        "TREx/RGB/stream0",
        [
            "20210205_0600_gill_rgb-04_full.h5",
            "20210205_0601_gill_rgb-04_full.h5",
            "20210205_0602_gill_rgb-04_full.h5",
            "20210205_0603_gill_rgb-04_full.h5",
            "20210205_0604_gill_rgb-04_full.h5",
            "20230101_0600_pina_rgb-02_full.h5",
        ],
    ),
    "read_trex_rgb/stream0.burst": (
        "TREx/RGB/stream0.burst",
        [
            "20211030_0600_gill_rgb-04_burst.png.tar",
            "20211030_0601_gill_rgb-04_burst.png.tar",
            "20211030_0602_gill_rgb-04_burst.png.tar",
            "20211030_0605_gill_rgb-04_burst.png.tar",
        ],
    ),
    "read_trex_spectrograph/stream0": (
        "TREx/spectrograph/l0/raw",
        [
            "20230101_0035_luck_spect-02_spectra_dark.pgm.gz",
            "20230101_0600_rabb_spect-01_spectra.pgm.gz",
            "20230503_0600_luck_spect-02_spectra.pgm.gz",
            "20230503_0601_luck_spect-02_spectra.pgm.gz",
            "20230503_0602_luck_spect-02_spectra.pgm.gz",
            "20230503_0603_luck_spect-02_spectra.pgm.gz",
            "20230503_0604_luck_spect-02_spectra.pgm.gz",
            "20230503_0605_luck_spect-02_spectra.pgm.gz",
        ],
    ),
}

# grid data: <data tree>/<path>/grid_files/MOSv001/<yyyy>/<mm>/<dd>/ut<hh>/<filename>
GRID_FILES = {
    "read_grid/multi_channel": (
        "TREx/RGB",
        [
            "20230324_0149_110km_MOSv001_grid_trex-rgb.h5",
            "20230324_0600_110km_MOSv001_grid_trex-rgb.h5",
            "20230324_0601_110km_MOSv001_grid_trex-rgb.h5",
            "20230324_0602_110km_MOSv001_grid_trex-rgb.h5",
            "20230324_0603_110km_MOSv001_grid_trex-rgb.h5",
            "20230324_0604_110km_MOSv001_grid_trex-rgb.h5",
            "20230324_0605_110km_MOSv001_grid_trex-rgb.h5",
        ],
    ),
    "read_grid/single_channel": (
        "THEMIS/asi",
        [
            "20230324_0019_110km_MOSv001_grid_themis-asi.h5",
            "20230324_0600_110km_MOSv001_grid_themis-asi.h5",
            "20230324_0601_110km_MOSv001_grid_themis-asi.h5",
            "20230324_0602_110km_MOSv001_grid_themis-asi.h5",
            "20230324_0603_110km_MOSv001_grid_themis-asi.h5",
            "20230324_0604_110km_MOSv001_grid_themis-asi.h5",
            "20230324_0605_110km_MOSv001_grid_themis-asi.h5",
        ],
    ),
}

# processed spectrograph data: <data tree>/<path>/<yyyy>/<mm>/<dd>/<site>_<device>/<filename>
SPECTROGRAPH_PROCESSED_FILES_PATH = "TREx/spectrograph/l1/processed"
SPECTROGRAPH_PROCESSED_FILES = [
    "20230503_05_rabb_spect-01_cal_v01.h5",
    "20230503_06_rabb_spect-01_cal_v01.h5",
    "20230503_07_rabb_spect-01_cal_v01.h5",
]

# SWAN HSR data: <data tree>/<path>/<yyyy>/<mm>/<dd>/<filename>
SWAN_HSR_FILES_PATH = "SWAN/hsr/l0/multi_freq/h5"
SWAN_HSR_FILES = [
    "20240203_mean-hsr_k0_v01.h5",
    "20240204_mean-hsr_k0_v01.h5",
    "20240205_mean-hsr_k0_v01.h5",
]

# riometer data: <data tree>/<path>/<yyyy>/<mm>/<dd>/<filename>
RIOMETER_K0_FILES_PATH = "GO-Canada/GO-Rio/txt"
RIOMETER_K0_FILES = [
    "chu_canopus_dcp_rio_20050402_v0.txt",
    "norstar_k0_rio-chur_20180501_v01.txt",
    "norstar_k0_rio-chur_20180502_v01.txt",
    "norstar_k0_rio-chur_20180503_v01.txt",
    "norstar_k0_rio-daws_20070402_v01.txt",
    "norstar_k0_rio-fsim_20180503_v01.txt",
    "norstar_k0_rio-isll_20140202_v01.txt",
]

# riometer data (k2): <data tree>/<path>/<yyyy>/<mm>/<dd>/<filename>
RIOMETER_K2_FILES_PATH = "GO-Canada/GO-Rio/txt"
RIOMETER_K2_FILES = [
    "chu_rio_19930403_v1a.txt",
    "daw_rio_19930403_v1a.txt",
    "daw_rio_19980304_v1a.txt",
    "mcm_rio_20030806_v1a.txt",
    "norstar_k2_rio-chur_20070402_v01.txt",
    "norstar_k2_rio-chur_20200501_v03.txt",
    "norstar_k2_rio-chur_20200502_v03.txt",
    "norstar_k2_rio-chur_20200503_v03.txt",
    "norstar_k2_rio-fsmi_20120504_v02.txt",
    "norstar_k2_rio-fsmi_20140202_v03.txt",
    "norstar_k2_rio-pina_20120504_v03.txt",
    "pin_rio_20050402_v1a.txt",
    "rab_rio_19900402_v1a.txt",
]

# REGO calibration data: <data tree>/<path>/<filename>
REGO_CALIBRATION_FILES_PATH = "GO-Canada/REGO/calibration"
REGO_CALIBRATION_FILES = [
    "REGO_Rayleighs_15649_20141015-20211018_v01.sav",
    "REGO_Rayleighs_15649_20211019-+_v02.sav",
    "REGO_Rayleighs_15651_20210908-+_v02.sav",
    "REGO_flatfield_15649_20211019-+_v02.sav",
    "REGO_flatfield_15653_20141002-20191212_v01.sav",
]

# skymaps: <data tree>/<path>/<filename>
SKYMAP_FILES = {
    "nir_skymap_atha_20220920-+_v01.sav": "TREx/NIR/skymaps/atha/atha_20220920",
    "rego_skymap_atha_20140718-+_v01.sav": "GO-Canada/REGO/skymap/atha/atha_20140718",
    "rgb_skymap_atha_20231003-+_v01.sav": "TREx/RGB/skymaps/atha/atha_20231003",
    "smile_skymap_atha_20241103-+_v01.sav": "SMILE/asi/l0/skymaps/atha/atha_20241103",
    "spect_skymap_luck_20230424-+_v01.sav": "TREx/spectrograph/skymaps/luck/luck_20230424",
    "themis_skymap_atha_20070301-20090522_vXX.sav": "THEMIS/asi/skymaps/atha/atha_20081029",
    "themis_skymap_atha_20230115-+_v02.sav": "THEMIS/asi/skymaps/atha/atha_20230115",
}

# the REGO skymap for luck that the test suite reads; the archive no longer has a skymap
# generated on this date, so the closest one to it is used, renamed to what the tests expect
RENAMED_SKYMAP_FILE = ("read_skymap/rego_skymap_luck_20230707-+_v01.sav",
                       "GO-Canada/REGO/skymap/luck/luck_20230606/rego_skymap_luck_20230606-+_v01.sav")

# TREx RGB files in the legacy PGM format, regenerated from the HDF5 file for the same minute
LEGACY_RGB_PGM_FILES = {
    "read_trex_rgb/unstable/stream0/20210503_0600_luck_rgb-03_full.pgm.gz": "2021/05/03/luck_rgb-03/ut06/20210503_0600_luck_rgb-03_full.h5",
    "read_trex_rgb/unstable/stream0/20210503_0601_luck_rgb-03_full.pgm.gz": "2021/05/03/luck_rgb-03/ut06/20210503_0601_luck_rgb-03_full.h5",
    "read_trex_rgb/unstable/stream0/20210503_0602_luck_rgb-03_full.pgm.gz": "2021/05/03/luck_rgb-03/ut06/20210503_0602_luck_rgb-03_full.h5",
    "read_trex_rgb/unstable/stream0/20210503_0603_luck_rgb-03_full.pgm.gz": "2021/05/03/luck_rgb-03/ut06/20210503_0603_luck_rgb-03_full.h5",
    "read_trex_rgb/unstable/stream0/20210503_0604_luck_rgb-03_full.pgm.gz": "2021/05/03/luck_rgb-03/ut06/20210503_0604_luck_rgb-03_full.h5",
    "read_trex_rgb/unstable/stream0/20210503_0605_luck_rgb-03_full.pgm.gz": "2021/05/03/luck_rgb-03/ut06/20210503_0605_luck_rgb-03_full.h5",
    "read_trex_rgb/unstable/stream0/20221226_1300_fsmi_rgb-01_full.pgm.gz": "2022/12/26/fsmi_rgb-01/ut13/20221226_1300_fsmi_rgb-01_full.h5",
}

# TREx RGB files in the legacy PNG tarball format, regenerated from the HDF5 file for the same minute
LEGACY_RGB_PNG_TAR_FILES = {
    "read_trex_rgb/unstable/stream0.colour/20200508_0600_gill_rgb-04_full.png.tar": "2020/05/08/gill_rgb-04/ut06/20200508_0600_gill_rgb-04_full.h5",
    "read_trex_rgb/unstable/stream0.colour/20200508_0601_gill_rgb-04_full.png.tar": "2020/05/08/gill_rgb-04/ut06/20200508_0601_gill_rgb-04_full.h5",
    "read_trex_rgb/unstable/stream0.colour/20200508_0602_gill_rgb-04_full.png.tar": "2020/05/08/gill_rgb-04/ut06/20200508_0602_gill_rgb-04_full.h5",
    "read_trex_rgb/unstable/stream0.colour/20200508_0603_gill_rgb-04_full.png.tar": "2020/05/08/gill_rgb-04/ut06/20200508_0603_gill_rgb-04_full.h5",
    "read_trex_rgb/unstable/stream0.colour/20200508_0604_gill_rgb-04_full.png.tar": "2020/05/08/gill_rgb-04/ut06/20200508_0604_gill_rgb-04_full.h5",
    "read_trex_rgb/unstable/stream0.colour/20200508_0605_gill_rgb-04_full.png.tar": "2020/05/08/gill_rgb-04/ut06/20200508_0605_gill_rgb-04_full.h5",
    "read_trex_rgb/unstable/stream0.colour/20220512_0700_fsmi_rgb-01_full.png.tar": "2022/05/12/fsmi_rgb-01/ut07/20220512_0700_fsmi_rgb-01_full.h5",
}

# the TREx RGB burst file captured in the 720p 'mode-b3' mode; the archive's copy of this
# minute has since been reprocessed down to the nominal resolution, so the 720p frames are
# regenerated from it
BURST_720P_FILE = ("read_trex_rgb/stream0.burst/20181208_1308_fsmi_rgb-01_mode-b3_raw.png.tar",
                   "TREx/RGB/stream0.burst/2018/12/08/fsmi_rgb-01/ut13/20181208_1308_fsmi_rgb-01_burst.png.tar")
BURST_720P_HEIGHT = 720
BURST_720P_WIDTH = 830

# the REGO file with a malformed metadata line, which the reader warns about but reads
# anyway; the archive's copy of this file has since been fixed
WARNING_FILE = ("read_rego/20170101_0600_resu_rego-655_6300.pgm.gz",
                "GO-Canada/REGO/stream0/2017/01/01/resu_rego-655/ut06/20170101_0600_resu_rego-655_6300.pgm.gz")

# files made by decompressing another file in the tree; each one is written out beside
# the file it came from, with the .gz suffix dropped
DECOMPRESSED_FILES = [
    "read_rego/20180403_0605_gill_rego-652_6300.pgm.gz",
    "read_themis/20140310_0605_gill_themis19_full.pgm.gz",
    "read_trex_blue/20220308_0605_gill_blue-814_full.pgm.gz",
    "read_trex_nir/20220307_0605_gill_nir-216_8446.pgm.gz",
    "read_trex_rgb/unstable/stream0/20210503_0605_luck_rgb-03_full.pgm.gz",
    "read_trex_spectrograph/stream0/20230503_0605_luck_spect-02_spectra.pgm.gz",
]

# single frames pulled out of a tarball of PNGs
#
# NOTE: order is --> destination: (source tarball, member filename)
TAR_MEMBER_FILES = {
    "read_trex_rgb/stream0.burst/20211030_060500_149606_gill_rgb-04_320ms_burst.png": (
        "read_trex_rgb/stream0.burst/20211030_0605_gill_rgb-04_burst.png.tar",
        "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
    ),
    "read_trex_rgb/unstable/stream0.colour/20200508_060500_122643_gill_rgb-04_320ms_full.png": (
        "read_trex_rgb/unstable/stream0.colour/20200508_0605_gill_rgb-04_full.png.tar",
        "20200508_060500_122643_gill_rgb-04_320ms_full.png",
    ),
}

# copies of another file in the tree; most of them are what the tests that chmod a file
# to 000 and read it use
#
# NOTE: order is --> (directory, source filename, destination filename)
COPIED_FILES = [
    (
        "read_grid/multi_channel",
        "20230324_0605_110km_MOSv001_grid_trex-rgb.h5",
        "20230324_0605_110km_MOSv001_grid_trex-rgb_badperms.h5",
    ),
    (
        "read_grid/single_channel",
        "20230324_0605_110km_MOSv001_grid_themis-asi.h5",
        "20230324_0605_110km_MOSv001_grid_themis-asi_badperms.h5",
    ),
    (
        "read_norstar_riometer/k0",
        "norstar_k0_rio-chur_20180501_v01.txt",
        "norstar_k0_rio-chur_20180501_v01.badperms.txt",
    ),
    (
        "read_norstar_riometer/k0",
        "norstar_k0_rio-fsim_20180503_v01.txt",
        "norstar_k0_rio-fsim_201805_v01.txt",
    ),
    (
        "read_norstar_riometer/k2",
        "rab_rio_19900402_v1a.txt",
        "rab_rio_19900402_v1a.badperms.txt",
    ),
    (
        "read_skymap",
        "themis_skymap_atha_20230115-+_v02.sav",
        "themis_skymap_gill_20210308-+_v02.sav",
    ),
    (
        "read_smile",
        "20250315_0600_atha_smile-31_rgb-full.h5",
        "20250330_0600_atha_smile-31_rgb-full_badperms.h5",
    ),
    (
        "read_swan_hsr",
        "20240205_mean-hsr_k0_v01.h5",
        "20240205_mean-hsr_k0_v01.badperms.h5",
    ),
    (
        "read_themis",
        "20140310_0600_gill_themis19_full.pgm.gz",
        "20211226_1624_gako_themis20_full_badperms.pgm.gz",
    ),
    (
        "read_trex_blue",
        "20220308_0600_gill_blue-814_full.pgm.gz",
        "20240101_0600_atha_blue-633_full_badperms.pgm.gz",
    ),
    (
        "read_trex_nir",
        "20220307_0600_gill_nir-216_8446.pgm.gz",
        "20200101_0700_gill_nir-217_8446_badperms.pgm.gz",
    ),
    (
        "read_trex_rgb/stream0.burst",
        "20211030_060500_149606_gill_rgb-04_320ms_burst.png",
        "20211030_060500_149606_gill_rgb-04_320ms_burst_badperms.png",
    ),
    (
        "read_trex_rgb/stream0",
        "20230101_0600_pina_rgb-02_full.h5",
        "20230101_0600_pina_rgb-02_full_badperms.h5",
    ),
    (
        "read_trex_rgb/unstable/stream0.colour",
        "20220512_0700_fsmi_rgb-01_full.png.tar",
        "20220512_0700_fsmi_rgb-01_full_badperms.png.tar",
    ),
    (
        "read_trex_rgb/unstable/stream0",
        "20221226_1300_fsmi_rgb-01_full.pgm.gz",
        "20221226_1300_fsmi_rgb-01_full_badperms.pgm.gz",
    ),
    (
        "read_trex_spectrograph/processed",
        "20230503_06_rabb_spect-01_cal_v01.h5",
        "20230503_06_rabb_spect-01_cal_v01.badperms.h5",
    ),
    (
        "read_trex_spectrograph/stream0",
        "20230503_0600_luck_spect-02_spectra.pgm.gz",
        "20230101_0600_rabb_spect-01_spectra.badperms.pgm.gz",
    ),
]

# files that the test suite expects to fail reading, recreated as empty files
#
# NOTE: an empty file reads as having no image data, which is one of the three ways the
# files that the tests point at were broken. See TRUNCATED_FILES and SHORT_FRAME_FILES
# for the other two.
EMPTY_FILES = [
    "read_rego/20201003_0504_kakt_rego-798_6300.pgm.gz",
    "read_rego/20201004_0519_kakt_rego-798_6300_badperms.pgm.gz",
    "read_rego/20230118_0627_luck_rego-651_6300.pgm.gz",
    "read_themis/20151106_0037_tpas_themis05_full.pgm.gz",
    "read_themis/20210101_0332_gill_themis19_full.pgm.gz",
    "read_themis/20211226_1624_gako_themis20_full.pgm.gz",
    "read_trex_blue/20230101_0700_atha_blue-633_full.pgm.gz",
    "read_trex_nir/20191126_0600_gill_nir-217_8446.pgm.gz",
    "read_trex_rgb/stream0.burst/20191121_0901_rabb_rgb-05_burst.png.tar",
    "read_trex_rgb/stream0.burst/20211030_06_gill_rgb-04_burst.png.tar",
    "read_trex_rgb/stream0/20201118_0235_rabb_rgb-06_full.h5",
    "read_trex_rgb/stream0/20201118_1006_rabb_rgb-06_full.h5",
    "read_trex_rgb/stream0/20210205_06_gill_rgb-04_full.h5",
    "read_trex_rgb/unstable/stream0.colour/20200508_06_gill_rgb-04_full.png.tar",
    "read_trex_rgb/unstable/stream0.colour/20220512_0711_fsmi_rgb-01_full.png.tar",
    "read_trex_rgb/unstable/stream0/20210503_06_luck_rgb-03_full.pgm.gz",
    "read_trex_rgb/unstable/stream0/20221226_1306_fsmi_rgb-01_full.pgm.gz",
    "read_trex_spectrograph/processed/20230503_rabb_spect-01_cal_v01.h5",
    "read_trex_spectrograph/stream0/20210909_0322_rabb_spect-01_spectra.pgm.gz",
]

# files that the test suite expects to fail reading, recreated by cutting a valid file off
# partway through; the reader hits the end of the compressed stream while it is reading
# metadata, which is what the original files did
#
# NOTE: order is --> (destination, file to truncate, fraction of it to keep)
TRUNCATED_FILES = [
    (
        "read_rego/20161102_2254_resu_rego-655_6300.pgm.gz",
        "read_rego/20180403_0600_gill_rego-652_6300.pgm.gz",
        0.4,
    ),
    (
        "read_rego/20201004_0519_kakt_rego-798_6300.pgm.gz",
        "read_rego/20180403_0601_gill_rego-652_6300.pgm.gz",
        0.2,
    ),
    (
        "read_themis/20150729_0246_gbay_themis03_full.pgm.gz",
        "read_themis/20140310_0600_gill_themis19_full.pgm.gz",
        0.4,
    ),
    (
        "read_themis/20151204_0621_inuv_themis17_full.pgm.gz",
        "read_themis/20140310_0601_gill_themis19_full.pgm.gz",
        0.2,
    ),
    (
        "read_trex_spectrograph/stream0/20190930_0559_luck_spect-02_spectra.pgm.gz",
        "read_trex_spectrograph/stream0/20230503_0600_luck_spect-02_spectra.pgm.gz",
        0.3,
    ),
]

# files that the test suite expects to fail reading, recreated by cutting a valid file off
# partway through its last image frame; the reader gets through the metadata and then finds
# the image data is short, which is what the original files did
#
# NOTE: order is --> (destination, file to cut short)
SHORT_FRAME_FILES = [
    (
        "read_rego/20161103_2323_resu_rego-655_6300.pgm.gz",
        "read_rego/20180403_0602_gill_rego-652_6300.pgm.gz",
    ),
    (
        "read_themis/20151030_0711_tpas_themis05_full.pgm.gz",
        "read_themis/20140310_0602_gill_themis19_full.pgm.gz",
    ),
    (
        "read_themis/20210101_0332_gill_themis19_full.pgm",
        "read_themis/20140310_0603_gill_themis19_full.pgm.gz",
    ),
    (
        "read_themis/20230104_0606_talo_themis15_full.pgm",
        "read_themis/20140310_0604_gill_themis19_full.pgm.gz",
    ),
]

# the 'not a data file at all' file that each reader is pointed at
PLACEHOLDER_FILE_DIRS = [
    "read_grid/multi_channel",
    "read_grid/single_channel",
    "read_norstar_riometer/k0",
    "read_norstar_riometer/k2",
    "read_rego",
    "read_smile",
    "read_swan_hsr",
    "read_themis",
    "read_trex_blue",
    "read_trex_nir",
    "read_trex_rgb/stream0",
    "read_trex_rgb/stream0.burst",
    "read_trex_rgb/unstable/stream0",
    "read_trex_rgb/unstable/stream0.colour",
    "read_trex_spectrograph/processed",
    "read_trex_spectrograph/stream0",
]
PLACEHOLDER_FILE_CONTENTS = "This is not a data file. It is here so that the test suite has something to point a reader at that it can't read.\n"


def parse_filename(filename):
    """
    Pull the date, hour, site UID and device UID out of a data filename. Filenames are
    either <yyyymmdd>_<hhmm>_<site>_<device>_... or, for the sub-minute files,
    <yyyymmdd>_<hhmmss>_<microseconds>_<site>_<device>_...
    """
    filename_split = filename.split('_')
    date_str = filename_split[0]
    hour_str = filename_split[1][0:2]
    if (len(filename_split[1]) == 6):
        site_uid = filename_split[3]
        device_uid = filename_split[4]
    else:
        site_uid = filename_split[2]
        device_uid = filename_split[3]
    return {
        "year": date_str[0:4],
        "month": date_str[4:6],
        "day": date_str[6:8],
        "hour": hour_str,
        "site_uid": site_uid,
        "device_uid": device_uid,
    }


def build_archive_manifest():
    """
    Work out the URL of every file that gets downloaded from the open data platform,
    returned as a dictionary of destination path in the test data tree --> URL.
    """
    manifest = {}

    # raw imager data
    for dest_dir, (path, filenames) in IMAGER_FILES.items():
        for filename in filenames:
            f = parse_filename(filename)
            manifest["%s/%s" % (dest_dir, filename)] = "%s/%s/%s/%s/%s/%s_%s/ut%s/%s" % (DATA_TREE, path, f["year"], f["month"], f["day"],
                                                                                         f["site_uid"], f["device_uid"], f["hour"], filename)

    # grid data
    for dest_dir, (path, filenames) in GRID_FILES.items():
        for filename in filenames:
            f = parse_filename(filename)
            manifest["%s/%s" % (dest_dir, filename)] = "%s/%s/grid_files/MOSv001/%s/%s/%s/ut%s/%s" % (DATA_TREE, path, f["year"], f["month"],
                                                                                                      f["day"], f["hour"], filename)

    # processed spectrograph data
    for filename in SPECTROGRAPH_PROCESSED_FILES:
        f = parse_filename(filename)
        manifest["read_trex_spectrograph/processed/%s" %
                 (filename)] = "%s/%s/%s/%s/%s/%s_%s/%s" % (DATA_TREE, SPECTROGRAPH_PROCESSED_FILES_PATH, f["year"], f["month"], f["day"],
                                                            f["site_uid"], f["device_uid"], filename)

    # SWAN HSR data
    for filename in SWAN_HSR_FILES:
        f = parse_filename(filename)
        manifest["read_swan_hsr/%s" % (filename)] = "%s/%s/%s/%s/%s/%s" % (DATA_TREE, SWAN_HSR_FILES_PATH, f["year"], f["month"], f["day"], filename)

    # riometer data
    #
    # NOTE: the date is the second-last underscore-delimited part of the filename, for both
    # the current and the historical filenaming
    riometer_groups = [
        ("read_norstar_riometer/k0", RIOMETER_K0_FILES_PATH, RIOMETER_K0_FILES),
        ("read_norstar_riometer/k2", RIOMETER_K2_FILES_PATH, RIOMETER_K2_FILES),
    ]
    for dest_dir, path, filenames in riometer_groups:
        for filename in filenames:
            date_str = filename.split('_')[-2]
            manifest["%s/%s" % (dest_dir, filename)] = "%s/%s/%s/%s/%s/%s" % (DATA_TREE, path, date_str[0:4], date_str[4:6], date_str[6:8], filename)

    # REGO calibration data
    for filename in REGO_CALIBRATION_FILES:
        manifest["read_calibration/%s" % (filename)] = "%s/%s/%s" % (DATA_TREE, REGO_CALIBRATION_FILES_PATH, filename)

    # skymaps
    for filename, path in SKYMAP_FILES.items():
        manifest["read_skymap/%s" % (filename)] = "%s/%s/%s" % (DATA_TREE, path, filename)
    manifest[RENAMED_SKYMAP_FILE[0]] = "%s/%s" % (DATA_TREE, RENAMED_SKYMAP_FILE[1])

    return manifest


def download_file(url, output_filename):
    if (os.path.exists(output_filename) and os.path.getsize(output_filename) > 0):
        return 0
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pyucalgarysrs test data builder"})
    with urllib.request.urlopen(req, timeout=300) as r:  # nosec
        content_type = r.headers.get("Content-Type", "")
        data = r.read()

    # the data server redirects to the home page for files that don't exist, so a
    # page of HTML means we asked for something that isn't there
    if ("text/html" in content_type.lower()):
        raise IOError("Received a page of HTML instead of a file, the URL is likely no longer valid: %s" % (url))
    with open(output_filename, "wb") as fp:
        fp.write(data)
    return len(data)


def download_archive_files(data_dir, n_parallel):

    def do_download(item):
        dest, url = item
        try:
            return (dest, download_file(url, "%s/%s" % (data_dir, dest)), None)
        except Exception as e:
            return (dest, 0, str(e))

    archive_manifest = build_archive_manifest()
    print("[downloading] %d files from the open data platform" % (len(archive_manifest)))
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        results = list(executor.map(do_download, sorted(archive_manifest.items())))
    failures = [r for r in results if r[2] is not None]
    for f in failures:
        print("  failed: %s (%s)" % (f[0], f[2]))
    if (len(failures) > 0):
        raise IOError("Failed to download %d file(s)" % (len(failures)))
    print("[downloading] retrieved %.1f MB" % (sum(r[1] for r in results) / 1e6))


def download_legacy_rgb_sources(work_dir, n_parallel):

    def do_download(archive_path):
        url = "%s/TREx/RGB/stream0/%s" % (DATA_TREE, archive_path)
        output_filename = "%s/%s" % (work_dir, os.path.basename(archive_path))
        try:
            return (archive_path, download_file(url, output_filename), None)
        except Exception as e:
            return (archive_path, 0, str(e))

    source_files = sorted(set(list(LEGACY_RGB_PGM_FILES.values()) + list(LEGACY_RGB_PNG_TAR_FILES.values())))
    print("[downloading] %d HDF5 files to regenerate the legacy TREx RGB files from" % (len(source_files)))
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        results = list(executor.map(do_download, source_files))
    failures = [r for r in results if r[2] is not None]
    for f in failures:
        print("  failed: %s (%s)" % (f[0], f[2]))
    if (len(failures) > 0):
        raise IOError("Failed to download %d file(s)" % (len(failures)))


def read_legacy_rgb_source(work_dir, archive_path):
    """
    Read the HDF5 file for a minute of TREx RGB data, and return the image data along
    with the timestamp of each frame.
    """
    import numpy as np
    import pyucalgarysrs

    srs = pyucalgarysrs.PyUCalgarySRS()
    filename = "%s/%s" % (work_dir, os.path.basename(archive_path))
    data = srs.data.readers.read_trex_rgb(filename)
    if (len(data.problematic_files) > 0):  # pragma: nocover-ok
        raise IOError("Failed to read source file %s" % (filename))

    # the legacy files are a fixed 20 frames per minute, at a fixed 3-second cadence
    #
    # NOTE: we use the timestamps out of the source file, microseconds and all. They are
    # the timestamps the legacy files were written with, so using them keeps the filenames
    # inside the regenerated tarballs identical to the originals.
    images = data.data
    if (images.shape[-1] < RGB_LEGACY_FRAMES):  # pragma: nocover-ok
        raise IOError("Source file %s has only %d frames" % (filename, images.shape[-1]))
    images = images[:, :, :, 0:RGB_LEGACY_FRAMES]
    if (images.shape[0] != RGB_LEGACY_HEIGHT or images.shape[1] != RGB_LEGACY_WIDTH):  # pragma: nocover-ok
        raise IOError("Source file %s has unexpected dimensions %s" % (filename, str(images.shape)))
    timestamps = list(data.timestamp[0:RGB_LEGACY_FRAMES])
    expected_seconds = [i * RGB_LEGACY_CADENCE_SECONDS for i in range(0, RGB_LEGACY_FRAMES)]
    if ([t.second for t in timestamps] != expected_seconds):  # pragma: nocover-ok
        raise IOError("Source file %s is not on the expected %d-second cadence" % (filename, RGB_LEGACY_CADENCE_SECONDS))
    return np.asarray(images), timestamps


def write_legacy_rgb_pgm_files(data_dir, work_dir):
    import numpy as np

    print("[generating] %d legacy TREx RGB PGM files" % (len(LEGACY_RGB_PGM_FILES)))
    for dest, archive_path in sorted(LEGACY_RGB_PGM_FILES.items()):
        images, timestamps = read_legacy_rgb_source(work_dir, archive_path)
        filename_split = os.path.basename(dest).split('_')
        site_uid = filename_split[2]
        device_uid = filename_split[3]

        # the legacy PGM files hold the single-channel 16-bit readout, so scale the
        # 8-bit RGB imagery of the source file up into a 16-bit luminance frame
        frames = images.mean(axis=2).astype("uint16") * 257

        output_bytes = bytearray()
        for i in range(0, RGB_LEGACY_FRAMES):
            timestamp_str = timestamps[i].strftime("%Y-%m-%d %H:%M:%S.%f UTC")
            metadata_lines = [
                ("Project unique ID", "trex"),
                ("Site unique ID", site_uid),
                ("Imager unique ID", device_uid),
                ("Mode unique ID", "full"),
                ("Image request start", timestamp_str),
                ("Exposure options", "WIDTH=%d HEIGHT=%d XBIN=1 YBIN=1 MSEC=320.00" % (RGB_LEGACY_WIDTH, RGB_LEGACY_HEIGHT)),
                #
                # NOTE: the real files repeat the temperature sensor line once per sensor, which
                # the reader gathers up into a list, so the regenerated files repeat it too
                ("Digitemp camera", "20.62 C"),
                ("Digitemp camera", "21.31 C"),
                ("Digitemp camera", "19.87 C"),
                ("Effective image exposure", "320.000 ms"),
            ]
            output_bytes += b"P5\n"
            for key, value in metadata_lines:
                output_bytes += ("#\"%s\" %s\n" % (key, value)).encode("ascii")
            output_bytes += ("%d %d\n65535\n" % (RGB_LEGACY_WIDTH, RGB_LEGACY_HEIGHT)).encode("ascii")
            output_bytes += frames[:, :, i].astype(np.dtype("uint16").newbyteorder('>')).tobytes()

        output_filename = "%s/%s" % (data_dir, dest)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with gzip.open(output_filename, "wb") as fp:
            fp.write(bytes(output_bytes))


def write_legacy_rgb_png_tar_files(data_dir, work_dir):
    import cv2

    print("[generating] %d legacy TREx RGB PNG tarballs" % (len(LEGACY_RGB_PNG_TAR_FILES)))
    for dest, archive_path in sorted(LEGACY_RGB_PNG_TAR_FILES.items()):
        images, timestamps = read_legacy_rgb_source(work_dir, archive_path)
        filename_split = os.path.basename(dest).split('_')
        site_uid = filename_split[2]
        device_uid = filename_split[3]

        # write each frame out as a PNG, then bundle them into the tarball
        frames_dir = "%s/%s" % (work_dir, os.path.basename(dest).replace(".png.tar", "_frames"))
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)
        png_filenames = []
        for i in range(0, RGB_LEGACY_FRAMES):
            png_filename = "%s_%s_%s_320ms_full.png" % (
                timestamps[i].strftime("%Y%m%d_%H%M%S_%f"),
                site_uid,
                device_uid,
            )
            cv2.imwrite("%s/%s" % (frames_dir, png_filename), cv2.cvtColor(images[:, :, :, i], cv2.COLOR_RGB2BGR))
            png_filenames.append(png_filename)

        output_filename = "%s/%s" % (data_dir, dest)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with tarfile.open(output_filename, "w") as tf:
            for png_filename in sorted(png_filenames):
                tf.add("%s/%s" % (frames_dir, png_filename), arcname=png_filename)
        shutil.rmtree(frames_dir, ignore_errors=True)


def write_burst_720p_file(data_dir, work_dir):
    import cv2
    import numpy as np

    dest, archive_path = BURST_720P_FILE
    print("[generating] the 720p TREx RGB burst file")
    source_filename = "%s/%s" % (work_dir, os.path.basename(archive_path))
    download_file("%s/%s" % (DATA_TREE, archive_path), source_filename)

    frames_dir = "%s/burst_720p_frames" % (work_dir)
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    png_filenames = []
    with tarfile.open(source_filename) as tf:
        for member in sorted(tf.getnames()):
            fp_in = tf.extractfile(member)
            if (fp_in is None):  # pragma: nocover-ok
                raise IOError("Member %s not found in %s" % (member, source_filename))
            image = cv2.imdecode(np.frombuffer(fp_in.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (BURST_720P_WIDTH, BURST_720P_HEIGHT), interpolation=cv2.INTER_CUBIC)
            png_filename = member.replace("_burst.png", "_mode-b3.png")
            cv2.imwrite("%s/%s" % (frames_dir, png_filename), image)
            png_filenames.append(png_filename)

    output_filename = "%s/%s" % (data_dir, dest)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with tarfile.open(output_filename, "w") as tf:
        for png_filename in sorted(png_filenames):
            tf.add("%s/%s" % (frames_dir, png_filename), arcname=png_filename)
    shutil.rmtree(frames_dir, ignore_errors=True)


def write_warning_file(data_dir, work_dir):
    dest, archive_path = WARNING_FILE
    print("[generating] the REGO file with a malformed metadata line")
    source_filename = "%s/%s" % (work_dir, os.path.basename(archive_path))
    download_file("%s/%s" % (DATA_TREE, archive_path), source_filename)

    # add a stray quote to the value of one metadata line, which is what the reader
    # warns about; everything else about the file is left alone
    with gzip.open(source_filename, "rb") as fp:
        file_bytes = fp.read()
    mangled = False
    output_lines = []
    for line in file_bytes.split(b"\n"):
        if (mangled is False and line.startswith(b'#"Site name"')):
            line = line + b' "'
            mangled = True
        output_lines.append(line)
    if (mangled is False):  # pragma: nocover-ok
        raise IOError("Did not find the metadata line to mangle in %s" % (source_filename))
    output_filename = "%s/%s" % (data_dir, dest)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with gzip.open(output_filename, "wb") as fp:
        fp.write(b"\n".join(output_lines))


def read_maybe_gzipped(filename):
    if (filename.endswith(".gz")):
        with gzip.open(filename, "rb") as fp:
            return fp.read()
    with open(filename, "rb") as fp:
        return fp.read()


def write_maybe_gzipped(filename, file_bytes):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if (filename.endswith(".gz")):
        with gzip.open(filename, "wb") as fp:
            fp.write(file_bytes)
    else:
        with open(filename, "wb") as fp:
            fp.write(file_bytes)


def write_truncated_files(data_dir):
    print("[generating] %d truncated files" % (len(TRUNCATED_FILES)))
    for dest, source, keep_fraction in TRUNCATED_FILES:
        source_filename = "%s/%s" % (data_dir, source)
        keep_bytes = int(os.path.getsize(source_filename) * keep_fraction)
        with open(source_filename, "rb") as fp:
            file_bytes = fp.read(keep_bytes)
        output_filename = "%s/%s" % (data_dir, dest)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "wb") as fp:
            fp.write(file_bytes)


def write_short_frame_files(data_dir):
    print("[generating] %d files that end partway through an image frame" % (len(SHORT_FRAME_FILES)))
    for dest, source in SHORT_FRAME_FILES:
        file_bytes = read_maybe_gzipped("%s/%s" % (data_dir, source))

        # cut the file off just inside the image data of its last frame; the image data
        # always follows the line holding the max pixel value
        last_frame_idx = file_bytes.rfind(SHORT_FRAME_MARKER)
        if (last_frame_idx == -1):  # pragma: nocover-ok
            raise IOError("Did not find the start of an image frame in %s" % (source))
        cutoff = last_frame_idx + len(SHORT_FRAME_MARKER) + SHORT_FRAME_KEEP_BYTES
        if (cutoff >= len(file_bytes)):  # pragma: nocover-ok
            raise IOError("The last image frame in %s is too small to cut short" % (source))
        write_maybe_gzipped("%s/%s" % (data_dir, dest), file_bytes[0:cutoff])


def write_derived_files(data_dir):
    print("[generating] %d decompressed files" % (len(DECOMPRESSED_FILES)))
    for source in DECOMPRESSED_FILES:
        with gzip.open("%s/%s" % (data_dir, source), "rb") as fp_in:
            with open("%s/%s" % (data_dir, source.removesuffix(".gz")), "wb") as fp_out:
                shutil.copyfileobj(fp_in, fp_out)

    print("[generating] %d single frames extracted from burst tarballs" % (len(TAR_MEMBER_FILES)))
    for dest, (source, member) in sorted(TAR_MEMBER_FILES.items()):
        with tarfile.open("%s/%s" % (data_dir, source)) as tf:
            fp_in = tf.extractfile(member)
            if (fp_in is None):  # pragma: nocover-ok
                raise IOError("Member %s not found in %s" % (member, source))
            with open("%s/%s" % (data_dir, dest), "wb") as fp_out:
                shutil.copyfileobj(fp_in, fp_out)

    print("[generating] %d copied files" % (len(COPIED_FILES)))
    for this_dir, source, dest in COPIED_FILES:
        shutil.copy("%s/%s/%s" % (data_dir, this_dir, source), "%s/%s/%s" % (data_dir, this_dir, dest))

    print("[generating] %d empty files" % (len(EMPTY_FILES)))
    for dest in sorted(EMPTY_FILES):
        output_filename = "%s/%s" % (data_dir, dest)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "wb"):
            pass

    print("[generating] %d placeholder files" % (len(PLACEHOLDER_FILE_DIRS)))
    for this_dir in sorted(PLACEHOLDER_FILE_DIRS):
        output_filename = "%s/%s/some_unexpected_file.txt" % (data_dir, this_dir)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w") as fp:
            fp.write(PLACEHOLDER_FILE_CONTENTS)


def set_permissions(data_dir):
    for root, _, files in os.walk(data_dir):
        for f in files:
            os.chmod(os.path.join(root, f), 0o644)


def create_tarball(data_dir):
    output_dir = os.path.dirname(data_dir)
    tarball_filename = "%s/%s" % (output_dir, TARBALL_FILENAME)
    print("[packaging] creating %s" % (tarball_filename))
    if (os.path.exists(tarball_filename)):
        os.remove(tarball_filename)
    subprocess.run(  # nosec
        ["tar", "-C", output_dir, "--exclude", WORK_DIR_NAME, "-czf", tarball_filename,
         os.path.basename(data_dir)],
        check=True,
    )
    return tarball_filename


def clean_data_dir(data_dir):
    print("[cleaning] removing the existing test data")
    for entry in sorted(glob.glob("%s/read_*" % (data_dir))):
        shutil.rmtree(entry, ignore_errors=True)


def main():
    # args
    parser = argparse.ArgumentParser(description="Build the test data that the test suite reads")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory to build the test data tree in (default: %s)" % (DEFAULT_DATA_DIR))
    parser.add_argument("--work-dir", default=None, help="Directory to cache intermediate downloads in (default: <data-dir>/%s)" % (WORK_DIR_NAME))
    parser.add_argument("--n-parallel", type=int, default=5, help="Number of parallel downloads (default: 5)")
    parser.add_argument("--clean", action="store_true", help="Remove the existing test data before building")
    parser.add_argument("--tarball", action="store_true", help="Also package the tree up as %s, beside the data directory" % (TARBALL_FILENAME))
    args = parser.parse_args()

    # set up paths
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    work_dir = "%s/%s" % (data_dir, WORK_DIR_NAME) if (args.work_dir is None) else os.path.abspath(os.path.expanduser(args.work_dir))
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # build it
    #
    # NOTE: files that are already there are left alone, so an interrupted build only has
    # to fetch what it didn't get to the first time
    if (args.clean is True):
        clean_data_dir(data_dir)
    download_archive_files(data_dir, args.n_parallel)
    download_legacy_rgb_sources(work_dir, args.n_parallel)
    write_legacy_rgb_pgm_files(data_dir, work_dir)
    write_legacy_rgb_png_tar_files(data_dir, work_dir)
    write_burst_720p_file(data_dir, work_dir)
    write_warning_file(data_dir, work_dir)
    write_derived_files(data_dir)
    write_truncated_files(data_dir)
    write_short_frame_files(data_dir)
    set_permissions(data_dir)
    print("\nDone, the test data is in %s" % (data_dir))

    # package it up, if we were asked to
    if (args.tarball is True):
        tarball_filename = create_tarball(data_dir)
        print("Packaged %s (%.1f MB)" % (tarball_filename, os.path.getsize(tarball_filename) / 1e6))
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
