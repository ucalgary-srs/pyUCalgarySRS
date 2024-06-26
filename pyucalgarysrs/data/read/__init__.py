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
Functions for reading data for specific datasets.
"""

import datetime
import os
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from ._themis import read as func_read_themis
from ._rego import read as func_read_rego
from ._trex_nir import read as func_read_trex_nir
from ._trex_blue import read as func_read_trex_blue
from ._trex_rgb import read as func_read_trex_rgb
from ._trex_spectrograph import read as func_read_trex_spectrograph
from ._skymap import read as func_read_skymap
from ._calibration import read as func_read_calibration
from ._grid import read as func_read_grid
from ..classes import (
    Dataset,
    Data,
    ProblematicFile,
    Skymap,
    SkymapGenerationInfo,
    Calibration,
    CalibrationGenerationInfo,
    GridData,
    GridSourceInfoData,
)
from ...exceptions import SRSUnsupportedReadError, SRSError


class ReadManager:
    """
    The ReadManager object is initialized within every PyUCalgarySRS.data object. It 
    acts as a way to access the submodules and carry over configuration information in 
    the super class.
    """

    __VALID_THEMIS_READFILE_DATASETS = ["THEMIS_ASI_RAW"]
    __VALID_REGO_READFILE_DATASETS = ["REGO_RAW"]
    __VALID_TREX_NIR_READFILE_DATASETS = ["TREX_NIR_RAW"]
    __VALID_TREX_BLUE_READFILE_DATASETS = ["TREX_BLUE_RAW"]
    __VALID_TREX_RGB_READFILE_DATASETS = ["TREX_RGB_RAW_NOMINAL", "TREX_RGB_RAW_BURST"]
    __VALID_SKYMAP_READFILE_DATASETS = [
        "REGO_SKYMAP_IDLSAV",
        "THEMIS_ASI_SKYMAP_IDLSAV",
        "TREX_NIR_SKYMAP_IDLSAV",
        "TREX_RGB_SKYMAP_IDLSAV",
        "TREX_BLUE_SKYMAP_IDLSAV",
    ]
    __VALID_CALIBRATION_READFILE_DATASETS = [
        "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
        "REGO_CALIBRATION_FLATFIELD_IDLSAV",
        "TREX_NIR_CALIBRATION_RAYLEIGHS_IDLSAV",
        "TREX_NIR_CALIBRATION_FLATFIELD_IDLSAV",
        "TREX_BLUE_CALIBRATION_RAYLEIGHS_IDLSAV",
        "TREX_BLUE_CALIBRATION_FLATFIELD_IDLSAV",
    ]
    __VALID_GRID_READFILE_DATASETS = [
        "THEMIS_ASI_GRID_MOSV001",
        "THEMIS_ASI_GRID_MOSU001",
        "REGO_GRID_MOSV001",
        "TREX_RGB_GRID_MOSV001",
        "TREX_NIR_GRID_MOSV001",
        "TREX_BLUE_GRID_MOSV001",
        "TREX_RGB5577_GRID_MOSV001",
    ]

    def __init__(self):
        pass

    def list_supported_datasets(self) -> List[str]:
        """
        List the datasets which have file reading capabilities supported.

        Returns:
            A list of the dataset names with file reading support.
        """
        supported_datasets = []
        for var in dir(self):
            var_lower = var.lower()
            if ("valid" in var_lower and "readfile_datasets" in var_lower):
                for dataset in getattr(self, var):
                    supported_datasets.append(dataset)
        supported_datasets = sorted(supported_datasets)
        return supported_datasets

    def is_supported(self, dataset_name: str) -> bool:
        """
        Check if a given dataset has file reading support. 
        
        Not all datasets available in the UCalgary Space Remote Sensing Open Data Platform 
        have special readfile routines in this library. This is because some datasets are 
        in basic formats such as JPG or PNG, so unique functions aren't necessary. We leave 
        it up to the user to open these basic files in whichever way they prefer. Use the 
        `list_supported_read_datasets()` function to see all datasets that have special
        file reading functionality in this library.

        Args:
            dataset_name (str): 
                The dataset name to check if file reading is supported. This parameter 
                is required.
        
        Returns:
            Boolean indicating if file reading is supported.
        """
        supported_datasets = self.list_supported_datasets()
        if (dataset_name in supported_datasets):
            return True
        else:
            return False

    def read(self,
             dataset: Dataset,
             file_list: Union[List[str], List[Path], str, Path],
             n_parallel: int = 1,
             first_record: bool = False,
             no_metadata: bool = False,
             quiet: bool = False) -> Data:
        """
        Read in data files for a given dataset. Note that only one type of dataset's data
        should be read in using a single call.

        Args:
            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                required.
            
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
            
            first_record (bool): 
                Only read in the first record in each file. This is the same as the first_frame
                parameter in the themis-imager-readfile and trex-imager-readfile libraries, and
                is a read optimization if you only need one image per minute, as opposed to the
                full temporal resolution of data (e.g., 3sec cadence). This parameter is optional.
            
            no_metadata (bool): 
                Skip reading of metadata. This is a minor optimization if the metadata is not needed.
                Default is `False`. This parameter is optional.
            
            quiet (bool): 
                Do not print out errors while reading data files, if any are encountered. Any files
                that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Data` object. This parameter
                is optional.
        
        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSUnsupportedReadError: an unsupported dataset was used when
                trying to read files.
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered

        Notes:
        ---------
        For users who are familiar with the themis-imager-readfile and trex-imager-readfile
        libraries, the read function provides a near-identical usage. Further improvements have 
        been integrated, and those libraries are anticipated to be deprecated at some point in the
        future.
        """
        # verify dataset is valid
        if (dataset is None):
            raise SRSUnsupportedReadError("Must supply a dataset. If not know, please use the srs.data.readers.read_<specific_routine>() function")

        # read data using the appropriate readfile routine
        if (dataset.name in self.__VALID_THEMIS_READFILE_DATASETS):
            return self.read_themis(file_list,
                                    n_parallel=n_parallel,
                                    first_record=first_record,
                                    no_metadata=no_metadata,
                                    quiet=quiet,
                                    dataset=dataset)
        elif (dataset.name in self.__VALID_REGO_READFILE_DATASETS):
            return self.read_rego(file_list, n_parallel=n_parallel, first_record=first_record, no_metadata=no_metadata, quiet=quiet, dataset=dataset)
        elif (dataset.name in self.__VALID_TREX_NIR_READFILE_DATASETS):
            return self.read_trex_nir(file_list,
                                      n_parallel=n_parallel,
                                      first_record=first_record,
                                      no_metadata=no_metadata,
                                      quiet=quiet,
                                      dataset=dataset)
        elif (dataset.name in self.__VALID_TREX_BLUE_READFILE_DATASETS):
            return self.read_trex_blue(file_list,
                                       n_parallel=n_parallel,
                                       first_record=first_record,
                                       no_metadata=no_metadata,
                                       quiet=quiet,
                                       dataset=dataset)
        elif (dataset.name in self.__VALID_TREX_RGB_READFILE_DATASETS):
            return self.read_trex_rgb(file_list,
                                      n_parallel=n_parallel,
                                      first_record=first_record,
                                      no_metadata=no_metadata,
                                      quiet=quiet,
                                      dataset=dataset)
        elif (dataset.name in self.__VALID_SKYMAP_READFILE_DATASETS):
            return self.read_skymap(file_list, n_parallel=n_parallel, quiet=quiet, dataset=dataset)
        elif (dataset.name in self.__VALID_CALIBRATION_READFILE_DATASETS):
            return self.read_calibration(file_list, n_parallel=n_parallel, quiet=quiet, dataset=dataset)
        elif (dataset.name in self.__VALID_GRID_READFILE_DATASETS):
            return self.read_grid(file_list, n_parallel=n_parallel, quiet=quiet, dataset=dataset)
        else:
            raise SRSUnsupportedReadError("Dataset does not have a supported read function")

    def read_themis(self,
                    file_list: Union[List[str], List[Path], str, Path],
                    n_parallel: int = 1,
                    first_record: bool = False,
                    no_metadata: bool = False,
                    quiet: bool = False,
                    dataset: Optional[Dataset] = None) -> Data:
        """
        Read in THEMIS ASI raw data (stream0 full.pgm* files).

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
            
            first_record (bool): 
                Only read in the first record in each file. This is the same as the first_frame
                parameter in the themis-imager-readfile and trex-imager-readfile libraries, and
                is a read optimization if you only need one image per minute, as opposed to the
                full temporal resolution of data (e.g., 3sec cadence). This parameter is optional.
            
            no_metadata (bool): 
                Skip reading of metadata. This is a minor optimization if the metadata is not needed.
                Default is `False`. This parameter is optional.
            
            quiet (bool): 
                Do not print out errors while reading data files, if any are encountered. Any files
                that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Data` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        img, meta, problematic_files = func_read_themis(
            file_list,
            n_parallel=n_parallel,
            first_record=first_record,
            no_metadata=no_metadata,
            quiet=quiet,
        )

        # generate timestamp array
        timestamp_list = []
        if (no_metadata is False):
            for m in meta:
                timestamp_list.append(datetime.datetime.strptime(m["Image request start"], "%Y-%m-%d %H:%M:%S.%f UTC"))

        # convert to return type
        problematic_files_objs = []
        for p in problematic_files:
            problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
        ret_obj = Data(
            data=img,
            timestamp=timestamp_list,
            metadata=meta,
            problematic_files=problematic_files_objs,
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return ret_obj

    def read_rego(self,
                  file_list: Union[List[str], List[Path], str, Path],
                  n_parallel: int = 1,
                  first_record: bool = False,
                  no_metadata: bool = False,
                  quiet: bool = False,
                  dataset: Optional[Dataset] = None) -> Data:
        """
        Read in REGO raw data (stream0 pgm* files).

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
            
            first_record (bool): 
                Only read in the first record in each file. This is the same as the first_frame
                parameter in the themis-imager-readfile and trex-imager-readfile libraries, and
                is a read optimization if you only need one image per minute, as opposed to the
                full temporal resolution of data (e.g., 3sec cadence). This parameter is optional.
            
            no_metadata (bool): 
                Skip reading of metadata. This is a minor optimization if the metadata is not needed.
                Default is `False`. This parameter is optional.
            
            quiet (bool): 
                Do not print out errors while reading data files, if any are encountered. Any files
                that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Data` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        img, meta, problematic_files = func_read_rego(
            file_list,
            n_parallel=n_parallel,
            first_record=first_record,
            no_metadata=no_metadata,
            quiet=quiet,
        )

        # generate timestamp array
        timestamp_list = []
        if (no_metadata is False):
            for m in meta:
                timestamp_list.append(datetime.datetime.strptime(m["Image request start"], "%Y-%m-%d %H:%M:%S.%f UTC"))

        # convert to return type
        problematic_files_objs = []
        for p in problematic_files:
            problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
        ret_obj = Data(
            data=img,
            timestamp=timestamp_list,
            metadata=meta,
            problematic_files=problematic_files_objs,
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return ret_obj

    def read_trex_nir(self,
                      file_list: Union[List[str], List[Path], str, Path],
                      n_parallel: int = 1,
                      first_record: bool = False,
                      no_metadata: bool = False,
                      quiet: bool = False,
                      dataset: Optional[Dataset] = None) -> Data:
        """
        Read in TREx near-infrared (NIR) raw data (stream0 pgm* files).

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
            
            first_record (bool): 
                Only read in the first record in each file. This is the same as the first_frame
                parameter in the themis-imager-readfile and trex-imager-readfile libraries, and
                is a read optimization if you only need one image per minute, as opposed to the
                full temporal resolution of data (e.g., 3sec cadence). This parameter is optional.
            
            no_metadata (bool): 
                Skip reading of metadata. This is a minor optimization if the metadata is not needed.
                Default is `False`. This parameter is optional.
            
            quiet (bool): 
                Do not print out errors while reading data files, if any are encountered. Any files
                that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Data` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        img, meta, problematic_files = func_read_trex_nir(
            file_list,
            n_parallel=n_parallel,
            first_record=first_record,
            no_metadata=no_metadata,
            quiet=quiet,
        )

        # generate timestamp array
        timestamp_list = []
        if (no_metadata is False):
            for m in meta:
                timestamp_list.append(datetime.datetime.strptime(m["Image request start"], "%Y-%m-%d %H:%M:%S.%f UTC"))

        # convert to appropriate return type
        problematic_files_objs = []
        for p in problematic_files:
            problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
        ret_obj = Data(
            data=img,
            timestamp=timestamp_list,
            metadata=meta,
            problematic_files=problematic_files_objs,
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return ret_obj

    def read_trex_blue(self,
                       file_list: Union[List[str], List[Path], str, Path],
                       n_parallel: int = 1,
                       first_record: bool = False,
                       no_metadata: bool = False,
                       quiet: bool = False,
                       dataset: Optional[Dataset] = None) -> Data:
        """
        Read in TREx Blueline raw data (stream0 pgm* files).

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
            
            first_record (bool): 
                Only read in the first record in each file. This is the same as the first_frame
                parameter in the themis-imager-readfile and trex-imager-readfile libraries, and
                is a read optimization if you only need one image per minute, as opposed to the
                full temporal resolution of data (e.g., 3sec cadence). This parameter is optional.
            
            no_metadata (bool): 
                Skip reading of metadata. This is a minor optimization if the metadata is not needed.
                Default is `False`. This parameter is optional.
            
            quiet (bool): 
                Do not print out errors while reading data files, if any are encountered. Any files
                that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Data` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        img, meta, problematic_files = func_read_trex_blue(
            file_list,
            n_parallel=n_parallel,
            first_record=first_record,
            no_metadata=no_metadata,
            quiet=quiet,
        )

        # generate timestamp array
        timestamp_list = []
        if (no_metadata is False):
            for m in meta:
                timestamp_list.append(datetime.datetime.strptime(m["Image request start"], "%Y-%m-%d %H:%M:%S.%f UTC"))

        # convert to return type
        problematic_files_objs = []
        for p in problematic_files:
            problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
        ret_obj = Data(
            data=img,
            timestamp=timestamp_list,
            metadata=meta,
            problematic_files=problematic_files_objs,
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return ret_obj

    def read_trex_rgb(self,
                      file_list: Union[List[str], List[Path], str, Path],
                      n_parallel: int = 1,
                      first_record: bool = False,
                      no_metadata: bool = False,
                      quiet: bool = False,
                      dataset: Optional[Dataset] = None) -> Data:
        """
        Read in TREx RGB raw data (stream0 h5, stream0.burst png.tar, unstable stream0 and stream0.colour pgm* and png*).

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
            
            first_record (bool): 
                Only read in the first record in each file. This is the same as the first_frame
                parameter in the themis-imager-readfile and trex-imager-readfile libraries, and
                is a read optimization if you only need one image per minute, as opposed to the
                full temporal resolution of data (e.g., 3sec cadence). This parameter is optional.
            
            no_metadata (bool): 
                Skip reading of metadata. This is a minor optimization if the metadata is not needed.
                Default is `False`. This parameter is optional.
            
            quiet (bool): 
                Do not print out errors while reading data files, if any are encountered. Any files
                that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Data` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        img, meta, problematic_files = func_read_trex_rgb(
            file_list,
            n_parallel=n_parallel,
            first_record=first_record,
            no_metadata=no_metadata,
            quiet=quiet,
        )

        # generate timestamp array
        timestamp_list = []
        if (no_metadata is False):
            for m in meta:
                if ("image_request_start_timestamp" in m):
                    timestamp_list.append(datetime.datetime.strptime(m["image_request_start_timestamp"], "%Y-%m-%d %H:%M:%S.%f UTC"))
                elif ("Image request start" in m):
                    timestamp_list.append(datetime.datetime.strptime(m["Image request start"], "%Y-%m-%d %H:%M:%S.%f UTC"))
                else:
                    raise SRSError("Unexpected timestamp metadata format")

        # convert to return type
        problematic_files_objs = []
        for p in problematic_files:
            problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
        ret_obj = Data(
            data=img,
            timestamp=timestamp_list,
            metadata=meta,
            problematic_files=problematic_files_objs,
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return ret_obj

    def read_trex_spectrograph(self,
                               file_list: Union[List[str], List[Path], str, Path],
                               n_parallel: int = 1,
                               first_record: bool = False,
                               no_metadata: bool = False,
                               quiet: bool = False,
                               dataset: Optional[Dataset] = None) -> Data:
        """
        Read in TREx Spectrograph raw data (stream0 pgm* files).

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
            
            first_record (bool): 
                Only read in the first record in each file. This is the same as the first_frame
                parameter in the themis-imager-readfile and trex-imager-readfile libraries, and
                is a read optimization if you only need one image per minute, as opposed to the
                full temporal resolution of data (e.g., 3sec cadence). This parameter is optional.
            
            no_metadata (bool): 
                Skip reading of metadata. This is a minor optimization if the metadata is not needed.
                Default is `False`. This parameter is optional.
            
            quiet (bool): 
                Do not print out errors while reading data files, if any are encountered. Any files
                that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Data` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        img, meta, problematic_files = func_read_trex_spectrograph(
            file_list,
            n_parallel=n_parallel,
            first_record=first_record,
            no_metadata=no_metadata,
            quiet=quiet,
        )

        # generate timestamp array
        timestamp_list = []
        if (no_metadata is False):
            for m in meta:
                timestamp_list.append(datetime.datetime.strptime(m["Image request start"], "%Y-%m-%d %H:%M:%S.%f UTC"))

        # convert to return type
        problematic_files_objs = []
        for p in problematic_files:
            problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
        ret_obj = Data(
            data=img,
            timestamp=timestamp_list,
            metadata=meta,
            problematic_files=problematic_files_objs,
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return ret_obj

    def read_skymap(
        self,
        file_list: Union[List[str], List[Path], str, Path],
        n_parallel: int = 1,
        quiet: bool = False,
        dataset: Optional[Dataset] = None,
    ) -> Data:
        """
        Read in UCalgary skymap files.

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
                                    
            quiet (bool): 
                Do not print out errors while reading skymap files, if any are encountered. Any 
                files that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Skymap` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        data = func_read_skymap(
            file_list,
            n_parallel=n_parallel,
            quiet=quiet,
        )

        # convert to return object
        skymap_objs = []
        for item in data:
            # init item
            item_recarray = item["skymap"][0]

            # parse valid start and end times into datetimes
            date_generated_dt = datetime.datetime.strptime(item_recarray.generation_info[0].date_generated.decode(), "%a %b %d %H:%M:%S %Y")

            # parse filename into several values
            filename_split = os.path.basename(item["filename"]).split('_')
            filename_times_split = filename_split[3].split('-')
            valid_interval_start_dt = datetime.datetime.strptime(filename_times_split[0], "%Y%m%d")
            valid_interval_stop_dt = None
            if (filename_times_split[1] != '+'):
                valid_interval_stop_dt = datetime.datetime.strptime(filename_times_split[1], "%Y%m%d")

            # parse date time used into datetime
            date_time_used_dt = datetime.datetime.strptime(item_recarray.generation_info[0].date_time_used.decode(), "%Y%m%d_UT%H")

            # determine the version
            version_str = os.path.splitext(item["filename"])[0].split('_')[-1]

            # create generation info dictionary
            generation_info_obj = SkymapGenerationInfo(
                author=item_recarray.generation_info[0].author.decode(),
                ccd_center=item_recarray.generation_info[0].ccd_center,
                code_used=item_recarray.generation_info[0].code_used.decode(),
                data_loc=item_recarray.generation_info[0].data_loc.decode(),
                date_generated=date_generated_dt,
                date_time_used=date_time_used_dt,
                img_flip=item_recarray.generation_info[0].img_flip,
                optical_orientation=item_recarray.generation_info[0].optical_orientation,
                optical_projection=item_recarray.generation_info[0].optical_projection,
                pixel_aspect_ratio=item_recarray.generation_info[0].pixel_aspect_ratio,
                valid_interval_start=valid_interval_start_dt,
                valid_interval_stop=valid_interval_stop_dt,
            )

            # add in bytscl_values parameter
            #
            # NOTE: bytscl_values was not present in early THEMIS skymap files, so
            # we conditionally add it
            if ("bytscl_values" in item_recarray.generation_info[0].dtype.names):
                generation_info_obj.bytscl_values = item_recarray.generation_info[0].bytscl_values

            # flip certain things
            full_elevation_flipped = np.flip(item_recarray.full_elevation, axis=0)
            full_azimuth_flipped = np.flip(item_recarray.full_azimuth, axis=0)
            full_map_latitude_flipped = np.flip(item_recarray.full_map_latitude, axis=1)
            full_map_longitude_flipped = np.flip(item_recarray.full_map_longitude, axis=1)
            if ("REGO" in item["filename"]):
                # flip e/w too, but just for REGO (since we do this to the raw data too)
                full_elevation_flipped = np.flip(full_elevation_flipped, axis=1)
                full_azimuth_flipped = np.flip(full_azimuth_flipped, axis=1)
                full_map_latitude_flipped = np.flip(full_map_latitude_flipped, axis=2)
                full_map_longitude_flipped = np.flip(full_map_longitude_flipped, axis=2)

            # create object
            skymap_obj = Skymap(
                filename=item["filename"],
                project_uid=item_recarray.project_uid.decode(),
                site_uid=item_recarray.site_uid.decode(),
                imager_uid=item_recarray.imager_uid.decode(),
                site_map_latitude=item_recarray.site_map_latitude,
                site_map_longitude=item_recarray.site_map_longitude,
                site_map_altitude=item_recarray.site_map_altitude,
                full_elevation=full_elevation_flipped,
                full_azimuth=full_azimuth_flipped,
                full_map_altitude=item_recarray.full_map_altitude,
                full_map_latitude=full_map_latitude_flipped,
                full_map_longitude=full_map_longitude_flipped,
                version=version_str,
                generation_info=generation_info_obj,
            )

            # append object
            skymap_objs.append(skymap_obj)

        # cast into data object
        data_obj = Data(
            data=skymap_objs,
            timestamp=[],
            metadata=[],
            problematic_files=[],
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return data_obj

    def read_calibration(
        self,
        file_list: Union[List[str], List[Path], str, Path],
        n_parallel: int = 1,
        quiet: bool = False,
        dataset: Optional[Dataset] = None,
    ) -> Data:
        """
        Read in UCalgary calibration files.

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.

            quiet (bool): 
                Do not print out errors while reading calibration files, if any are encountered. 
                Any files that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Calibration` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        data = func_read_calibration(
            file_list,
            n_parallel=n_parallel,
            quiet=quiet,
        )

        # convert to return object
        calibration_objs = []
        for item in data:
            # init
            item_filename = item["filename"]

            # determine the version
            version_str = os.path.splitext(item_filename)[0].split('_')[-1]

            # parse filename into several values
            filename_split = os.path.basename(item_filename).split('_')
            filename_times_split = filename_split[3].split('-')
            valid_interval_start_dt = datetime.datetime.strptime(filename_times_split[0], "%Y%m%d")
            valid_interval_stop_dt = None
            if (filename_times_split[1] != '+'):
                valid_interval_stop_dt = datetime.datetime.strptime(filename_times_split[1], "%Y%m%d")

            # determine the detector UID
            detector_uid = filename_split[2]
            file_type = filename_split[1].lower()
            flat_field_multiplier_value = None
            rayleighs_perdn_persecond_value = None
            if (file_type == "flatfield"):
                for key in item.keys():
                    if ("flat_field_multiplier" in key):
                        # flip vertically
                        flat_field_multiplier_value = np.flip(item[key], axis=0)

                        # flip horizontally, if REGO
                        if ("REGO" in item_filename):
                            flat_field_multiplier_value = np.flip(flat_field_multiplier_value, axis=1)
                        break
            elif (file_type == "rayleighs"):
                for key in item.keys():
                    if ("rper_dnpersecond" in key):
                        rayleighs_perdn_persecond_value = item[key]
                        break

            # set input data dir and skymap filename (may exist in the calibration file, may not)
            author_str = None
            input_data_dir_str = None
            skymap_filename_str = None
            if ("author" in item):
                author_str = item["author"].decode()
            if ("input_data_dir" in item):
                input_data_dir_str = item["input_data_dir"].decode()
            if ("skymap_filename" in item):
                skymap_filename_str = item["skymap_filename"].decode()

            # set generation info object
            generation_info_obj = CalibrationGenerationInfo(
                author=author_str,
                input_data_dir=input_data_dir_str,
                skymap_filename=skymap_filename_str,
                valid_interval_start=valid_interval_start_dt,
                valid_interval_stop=valid_interval_stop_dt,
            )

            # create object
            calibration_obj = Calibration(
                filename=item_filename,
                version=version_str,
                dataset=dataset,
                detector_uid=detector_uid,
                flat_field_multiplier=flat_field_multiplier_value,
                rayleighs_perdn_persecond=rayleighs_perdn_persecond_value,
                generation_info=generation_info_obj,
            )

            # append object
            calibration_objs.append(calibration_obj)

        # cast into data object
        data_obj = Data(
            data=calibration_objs,
            timestamp=[],
            metadata=[],
            problematic_files=[],
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return data_obj

    def read_grid(self,
                  file_list: Union[List[str], List[Path], str, Path],
                  n_parallel: int = 1,
                  first_record: bool = False,
                  no_metadata: bool = False,
                  quiet: bool = False,
                  dataset: Optional[Dataset] = None) -> Data:
        """
        Read in grid files.

        Args:
            file_list (List[str], List[Path], str, Path): 
                The files to read in. Absolute paths are recommended, but not technically
                necessary. This can be a single string for a file, or a list of strings to read
                in multiple files. This parameter is required.

            n_parallel (int): 
                Number of data files to read in parallel using multiprocessing. Default value 
                is 1. Adjust according to your computer's available resources. This parameter 
                is optional.
            
            first_record (bool): 
                Only read in the first record in each file. This is the same as the first_frame
                parameter in the themis-imager-readfile and trex-imager-readfile libraries, and
                is a read optimization if you only need one image per minute, as opposed to the
                full temporal resolution of data (e.g., 3sec cadence). This parameter is optional.
            
            no_metadata (bool): 
                Skip reading of metadata. This is a minor optimization if the metadata is not needed.
                Default is `False`. This parameter is optional.
            
            quiet (bool): 
                Do not print out errors while reading data files, if any are encountered. Any files
                that encounter errors will be, as usual, accessible via the `problematic_files` 
                attribute of the returned `pyucalgarysrs.data.classes.Data` object. This parameter
                is optional.

            dataset (pyucalgarysrs.data.classes.Dataset): 
                The dataset object for which the files are associated with. This parameter is
                optional.

        Returns:
            A `pyucalgarysrs.data.classes.Data` object containing the data read in, among other
            values.
        
        Raises:
            pyucalgarysrs.exceptions.SRSError: a generic read error was encountered
        """
        # read data
        data_dict, meta, problematic_files = func_read_grid(
            file_list,
            n_parallel=n_parallel,
            first_record=first_record,
            no_metadata=no_metadata,
            quiet=quiet,
        )

        # create grid data object
        grid_data_obj = GridData(
            grid=data_dict["grid"],  # type: ignore
            fill_value=float(meta[0]["fill_value"]),
            source_info=GridSourceInfoData(confidence=data_dict["source_info"]["confidence"]),  # type: ignore
        )

        # generate timestamp array
        timestamp_list = []
        if (no_metadata is False):
            for t in data_dict["timestamp"]:  # type: ignore
                timestamp_list.append(datetime.datetime.strptime(t.decode(), "%Y-%m-%d %H:%M:%S UTC"))

        # convert to return type
        problematic_files_objs = []
        for p in problematic_files:
            problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
        ret_obj = Data(
            data=grid_data_obj,
            timestamp=timestamp_list,
            metadata=meta,
            problematic_files=problematic_files_objs,
            calibrated_data=None,
            dataset=dataset,
        )

        # return
        return ret_obj
