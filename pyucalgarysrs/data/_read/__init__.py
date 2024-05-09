import datetime
from typing import List, Union, Optional
from ._themis import read as func_read_themis
from ._rego import read as func_read_rego
from ._trex_nir import read as func_read_trex_nir
from ._trex_blue import read as func_read_trex_blue
from ._trex_rgb import read as func_read_trex_rgb
from ._trex_spectrograph import read as func_read_trex_spectrograph
from .._schemas import Dataset, Data, ProblematicFile
from ...exceptions import SRSUnsupportedReadException, SRSException


class ReadManager:

    __VALID_THEMIS_READFILE_DATASETS = ["THEMIS_ASI_RAW"]
    __VALID_REGO_READFILE_DATASETS = ["REGO_RAW"]
    __VALID_TREX_NIR_READFILE_DATASETS = ["TREX_NIR_RAW"]
    __VALID_TREX_BLUE_READFILE_DATASETS = ["TREX_BLUE_RAW"]
    __VALID_TREX_RGB_READFILE_DATASETS = ["TREX_RGB_RAW_NOMINAL", "TREX_RGB_RAW_BURST"]

    def __init__(self):
        pass

    def list_supported_datasets(self) -> List[str]:
        """
        List the datasets which have file reading capabilities supported.
        """
        supported_datasets = []
        for var in dir(self):
            var_lower = var.lower()
            if ("valid" in var_lower and "readfile_datasets" in var_lower):
                for dataset in getattr(self, var):
                    supported_datasets.append(dataset)
        supported_datasets = sorted(supported_datasets)
        return supported_datasets

    def check_if_supported(self, dataset_name: str) -> bool:
        """
        Check if a given dataset has file reading support
        """
        supported_datasets = self.list_supported_datasets()
        if (dataset_name in supported_datasets):
            return True
        else:
            return False

    def read(self,
             dataset: Dataset,
             file_list: Union[List[str], str],
             n_parallel: int = 1,
             first_record: bool = False,
             no_metadata: bool = False,
             quiet: bool = False,
             as_xarray: bool = False):
        """
        This function reads in data, using the derived readfile based on the dataset name
        """
        # verify dataset is valid
        if (dataset is None):
            raise SRSUnsupportedReadException(
                "Must supply a dataset. If not know, please use the srs.data.readers.read_<specific_routine>() function")

        # read data using the appropriate readfile routine
        if (dataset.name in self.__VALID_THEMIS_READFILE_DATASETS):
            return self.read_themis(file_list,
                                    n_parallel=n_parallel,
                                    first_record=first_record,
                                    no_metadata=no_metadata,
                                    quiet=quiet,
                                    as_xarray=as_xarray,
                                    dataset=dataset)
        elif (dataset.name in self.__VALID_REGO_READFILE_DATASETS):
            return self.read_rego(file_list,
                                  n_parallel=n_parallel,
                                  first_record=first_record,
                                  no_metadata=no_metadata,
                                  quiet=quiet,
                                  as_xarray=as_xarray,
                                  dataset=dataset)
        elif (dataset.name in self.__VALID_TREX_NIR_READFILE_DATASETS):
            return self.read_trex_nir(file_list,
                                      n_parallel=n_parallel,
                                      first_record=first_record,
                                      no_metadata=no_metadata,
                                      quiet=quiet,
                                      as_xarray=as_xarray,
                                      dataset=dataset)
        elif (dataset.name in self.__VALID_TREX_BLUE_READFILE_DATASETS):
            return self.read_trex_blue(file_list,
                                       n_parallel=n_parallel,
                                       first_record=first_record,
                                       no_metadata=no_metadata,
                                       quiet=quiet,
                                       as_xarray=as_xarray,
                                       dataset=dataset)
        elif (dataset.name in self.__VALID_TREX_RGB_READFILE_DATASETS):
            return self.read_trex_rgb(file_list,
                                      n_parallel=n_parallel,
                                      first_record=first_record,
                                      no_metadata=no_metadata,
                                      quiet=quiet,
                                      as_xarray=as_xarray,
                                      dataset=dataset)
        else:
            raise SRSUnsupportedReadException("Dataset does not have a supported read function")

    def read_themis(self,
                    file_list: Union[List[str], str],
                    n_parallel: int = 1,
                    first_record: bool = False,
                    no_metadata: bool = False,
                    quiet: bool = False,
                    as_xarray: bool = False,
                    dataset: Optional[Dataset] = None) -> Union[None, Data]:
        """
        Read in THEMIS ASI raw data (stream0 full.pgm* files)
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

        # convert to appropriate return type
        if (as_xarray is True):
            ret_obj = None
        else:
            problematic_files_objs = []
            for p in problematic_files:
                problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
            ret_obj = Data(
                data=img,
                timestamp=timestamp_list,
                metadata=meta,
                problematic_files=problematic_files_objs,
                dataset=dataset,
            )

        # return
        return ret_obj

    def read_rego(self,
                  file_list: Union[List[str], str],
                  n_parallel: int = 1,
                  first_record: bool = False,
                  no_metadata: bool = False,
                  quiet: bool = False,
                  as_xarray: bool = False,
                  dataset: Optional[Dataset] = None) -> Union[None, Data]:
        """
        Read in REGO raw data (stream0 pgm* files)
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

        # convert to appropriate return type
        if (as_xarray is True):
            ret_obj = None
        else:
            problematic_files_objs = []
            for p in problematic_files:
                problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
            ret_obj = Data(
                data=img,
                timestamp=timestamp_list,
                metadata=meta,
                problematic_files=problematic_files_objs,
                dataset=dataset,
            )

        # return
        return ret_obj

    def read_trex_nir(self,
                      file_list: Union[List[str], str],
                      n_parallel: int = 1,
                      first_record: bool = False,
                      no_metadata: bool = False,
                      quiet: bool = False,
                      as_xarray: bool = False,
                      dataset: Optional[Dataset] = None) -> Union[None, Data]:
        """
        Read in TREx near-infrared (NIR) raw data (stream0 pgm* files)
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
        if (as_xarray is True):
            ret_obj = None
        else:
            problematic_files_objs = []
            for p in problematic_files:
                problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
            ret_obj = Data(
                data=img,
                timestamp=timestamp_list,
                metadata=meta,
                problematic_files=problematic_files_objs,
                dataset=dataset,
            )

        # return
        return ret_obj

    def read_trex_blue(self,
                       file_list: Union[List[str], str],
                       n_parallel: int = 1,
                       first_record: bool = False,
                       no_metadata: bool = False,
                       quiet: bool = False,
                       as_xarray: bool = False,
                       dataset: Optional[Dataset] = None) -> Union[None, Data]:
        """
        Read in TREx Blueline raw data (stream0 pgm* files)
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

        # convert to appropriate return type
        if (as_xarray is True):
            ret_obj = None
        else:
            problematic_files_objs = []
            for p in problematic_files:
                problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
            ret_obj = Data(
                data=img,
                timestamp=timestamp_list,
                metadata=meta,
                problematic_files=problematic_files_objs,
                dataset=dataset,
            )

        # return
        return ret_obj

    def read_trex_rgb(self,
                      file_list: Union[List[str], str],
                      n_parallel: int = 1,
                      first_record: bool = False,
                      no_metadata: bool = False,
                      quiet: bool = False,
                      as_xarray: bool = False,
                      dataset: Optional[Dataset] = None) -> Union[None, Data]:
        """
        Read in TREx RGB raw data (stream0 h5, stream0.burst png.tar, unstable stream0 and stream0.colour pgm* and png*)
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
                    raise SRSException("Unexpected timestamp metadata format")

        # convert to appropriate return type
        if (as_xarray is True):
            ret_obj = None
        else:
            problematic_files_objs = []
            for p in problematic_files:
                problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
            ret_obj = Data(
                data=img,
                timestamp=timestamp_list,
                metadata=meta,
                problematic_files=problematic_files_objs,
                dataset=dataset,
            )

        # return
        return ret_obj

    def read_trex_spectrograph(self,
                               file_list: Union[List[str], str],
                               n_parallel: int = 1,
                               first_record: bool = False,
                               no_metadata: bool = False,
                               quiet: bool = False,
                               as_xarray: bool = False,
                               dataset: Optional[Dataset] = None) -> Union[None, Data]:
        """
        Read in TREx Spectrograph raw data (stream0 pgm* files)
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

        # convert to appropriate return type
        if (as_xarray is True):
            ret_obj = None
        else:
            problematic_files_objs = []
            for p in problematic_files:
                problematic_files_objs.append(ProblematicFile(p["filename"], error_message=p["error_message"], error_type="error"))
            ret_obj = Data(
                data=img,
                timestamp=timestamp_list,
                metadata=meta,
                problematic_files=problematic_files_objs,
                dataset=dataset,
            )

        # return
        return ret_obj
