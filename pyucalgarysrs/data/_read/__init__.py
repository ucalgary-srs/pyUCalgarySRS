import datetime
from typing import List, Union, Optional
from ._themis import read as func_read_themis
from ._rego import read as func_read_rego
from ._trex_nir import read as func_read_trex_nir
from ._trex_blue import read as func_read_trex_blue
from ._trex_rgb import read as func_read_trex_rgb
from ._trex_spectrograph import read as func_read_trex_spectrograph
from ._skymaps import read as func_read_skymaps
from .._schemas import Dataset, Data, ProblematicFile, Skymap
from ...exceptions import SRSUnsupportedReadError, SRSError


class ReadManager:

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
        # "TREX_BLUE_SKYMAP_IDLSAV",
    ]
    __VALID_CALIBRATION_READFILE_DATASETS = [
        "REGO_CALIBRATION_RAYLEIGHS_IDLSAV",
        "REGO_CALIBRATION_FLATFIELD_IDLSAV",
        "TREX_NIR_CALIBRATION_RAYLEIGHS_IDLSAV",
        "TREX_NIR_CALIBRATION_FLATFIELD_IDLSAV",
    ]

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
             quiet: bool = False) -> Union[Data, List[Skymap]]:
        """
        This function reads in data, using the derived readfile based on the dataset name
        """
        # verify dataset is valid
        if (dataset is None):
            raise SRSUnsupportedReadError(
                "Must supply a dataset. If not know, please use the srs.data.readers.read_<specific_routine>() function")

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
            return self.read_skymaps(file_list, n_parallel=n_parallel, quiet=quiet, dataset=dataset)
        else:
            raise SRSUnsupportedReadError("Dataset does not have a supported read function")

    def read_themis(self,
                    file_list: Union[List[str], str],
                    n_parallel: int = 1,
                    first_record: bool = False,
                    no_metadata: bool = False,
                    quiet: bool = False,
                    dataset: Optional[Dataset] = None) -> Data:
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

        # convert to return type
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
                  dataset: Optional[Dataset] = None) -> Data:
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

        # convert to return type
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
                      dataset: Optional[Dataset] = None) -> Data:
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
                       dataset: Optional[Dataset] = None) -> Data:
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

        # convert to return type
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
                      dataset: Optional[Dataset] = None) -> Data:
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
                               dataset: Optional[Dataset] = None) -> Data:
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

        # convert to return type
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

    def read_skymaps(self,
                     file_list: Union[List[str], str],
                     n_parallel: int = 1,
                     quiet: bool = False,
                     dataset: Optional[Dataset] = None) -> List[Skymap]:
        """
        Read in UCalgary skymap files
        """
        # read data
        data = func_read_skymaps(
            file_list,
            n_parallel=n_parallel,
            quiet=quiet,
        )

        # convert to return object
        ret_list = []
        for item in data:
            item = item["skymap"][0]  # type: ignore
            # create generation info dictionary
            generation_info_dict = {
                "author": item.generation_info[0].author.decode(),
                "ccd_center": item.generation_info[0].ccd_center,
                "code_used": item.generation_info[0].code_used.decode(),
                "data_loc": item.generation_info[0].data_loc.decode(),
                "date_generated": item.generation_info[0].date_generated.decode(),
                "date_time_used": item.generation_info[0].date_time_used.decode(),
                "img_flip": item.generation_info[0].img_flip,
                "optical_orientation": item.generation_info[0].optical_orientation,
                "optical_projection": item.generation_info[0].optical_projection,
                "pixel_aspect_ratio": item.generation_info[0].pixel_aspect_ratio,
                "valid_interval_start": item.generation_info[0].valid_interval_start.decode(),
                "valid_interval_stop": item.generation_info[0].valid_interval_stop.decode(),
            }

            # add in bytscl_values parameter
            #
            # NOTE: bytscl_values was not present in early THEMIS skymap files, so
            # we conditionally add it
            if ("bytscl_values" in item.generation_info[0].dtype.names):
                generation_info_dict["bytscl_values"] = item.generation_info[0].bytscl_values

            # create object
            ret_obj = Skymap(
                project_uid=item.project_uid.decode(),
                site_uid=item.site_uid.decode(),
                imager_uid=item.imager_uid.decode(),
                site_map_latitude=item.site_map_latitude,
                site_map_longitude=item.site_map_longitude,
                site_map_altitude=item.site_map_altitude,
                bin_row=item.bin_row,
                bin_column=item.bin_column,
                bin_elevation=item.bin_elevation,
                bin_azimuth=item.bin_azimuth,
                bin_map_altitude=item.bin_map_altitude,
                bin_map_latitude=item.bin_map_latitude,
                bin_map_longitude=item.bin_map_longitude,
                full_row=item.full_row,
                full_column=item.full_column,
                full_ignore=item.full_ignore,
                full_subtract=item.full_subtract,
                full_multiply=item.full_multiply,
                full_elevation=item.full_elevation,
                full_azimuth=item.full_azimuth,
                full_map_altitude=item.full_map_altitude,
                full_map_latitude=item.full_map_latitude,
                full_map_longitude=item.full_map_longitude,
                full_bin=item.full_bin,
                generation_info=generation_info_dict,
                dataset=dataset,
            )

            # append object
            ret_list.append(ret_obj)

        # return
        return ret_list
