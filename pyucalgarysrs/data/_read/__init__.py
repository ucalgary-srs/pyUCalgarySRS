import datetime
import os
from typing import List, Union, Optional
from ._themis import read as func_read_themis
from ._rego import read as func_read_rego
from ._trex_nir import read as func_read_trex_nir
from ._trex_blue import read as func_read_trex_blue
from ._trex_rgb import read as func_read_trex_rgb
from ._trex_spectrograph import read as func_read_trex_spectrograph
from ._skymap import read as func_read_skymap
from ._calibration import read as func_read_calibration
from .._schemas import (
    Dataset,
    Data,
    ProblematicFile,
    Skymap,
    SkymapGenerationInfo,
    Calibration,
    CalibrationGenerationInfo,
)
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
             quiet: bool = False) -> Union[Data, List[Skymap], List[Calibration]]:
        """
        This function reads in data, using the derived readfile based on the dataset name
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

    def read_skymap(self,
                    file_list: Union[List[str], str],
                    n_parallel: int = 1,
                    quiet: bool = False,
                    dataset: Optional[Dataset] = None) -> List[Skymap]:
        """
        Read in UCalgary skymap files
        """
        # read data
        data = func_read_skymap(
            file_list,
            n_parallel=n_parallel,
            quiet=quiet,
        )

        # convert to return object
        ret_list = []
        for item in data:
            # init item
            item_recarray = item["skymap"][0]

            # parse several values into datetimes
            date_generated_dt = datetime.datetime.strptime(item_recarray.generation_info[0].date_generated.decode(), "%a %b %d %H:%M:%S %Y")
            valid_interval_start_dt = datetime.datetime(2000, 1, 1, 0, 0, 0)
            try:
                valid_interval_start_dt = datetime.datetime.strptime(item_recarray.generation_info[0].valid_interval_start.decode(), "%Y%m%d%H")
            except Exception:
                try:
                    valid_interval_start_dt = datetime.datetime.strptime(item_recarray.generation_info[0].valid_interval_start.decode(), "%Y%m%d")
                except Exception:
                    pass
            valid_interval_stop_dt = None
            if (item_recarray.generation_info[0].valid_interval_stop.decode() != "+"):
                try:
                    valid_interval_stop_dt = datetime.datetime.strptime(item_recarray.generation_info[0].valid_interval_stop.decode(), "%Y%m%d%H")
                except Exception:
                    try:
                        valid_interval_stop_dt = datetime.datetime.strptime(item_recarray.generation_info[0].valid_interval_stop.decode(), "%Y%m%d")
                    except Exception:
                        pass

            # determine the version
            version_str = os.path.splitext(item["filename"])[0].split('_')[-1]

            # create generation info dictionary
            generation_info_obj = SkymapGenerationInfo(
                author=item_recarray.generation_info[0].author.decode(),
                ccd_center=item_recarray.generation_info[0].ccd_center,
                code_used=item_recarray.generation_info[0].code_used.decode(),
                data_loc=item_recarray.generation_info[0].data_loc.decode(),
                date_generated=date_generated_dt,
                date_time_used=item_recarray.generation_info[0].date_time_used.decode(),
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

            # create object
            ret_obj = Skymap(
                filename=item["filename"],
                project_uid=item_recarray.project_uid.decode(),
                site_uid=item_recarray.site_uid.decode(),
                imager_uid=item_recarray.imager_uid.decode(),
                site_map_latitude=item_recarray.site_map_latitude,
                site_map_longitude=item_recarray.site_map_longitude,
                site_map_altitude=item_recarray.site_map_altitude,
                bin_row=item_recarray.bin_row,
                bin_column=item_recarray.bin_column,
                bin_elevation=item_recarray.bin_elevation,
                bin_azimuth=item_recarray.bin_azimuth,
                bin_map_altitude=item_recarray.bin_map_altitude,
                bin_map_latitude=item_recarray.bin_map_latitude,
                bin_map_longitude=item_recarray.bin_map_longitude,
                full_row=item_recarray.full_row,
                full_column=item_recarray.full_column,
                full_ignore=item_recarray.full_ignore,
                full_subtract=item_recarray.full_subtract,
                full_multiply=item_recarray.full_multiply,
                full_elevation=item_recarray.full_elevation,
                full_azimuth=item_recarray.full_azimuth,
                full_map_altitude=item_recarray.full_map_altitude,
                full_map_latitude=item_recarray.full_map_latitude,
                full_map_longitude=item_recarray.full_map_longitude,
                full_bin=item_recarray.full_bin,
                version=version_str,
                generation_info=generation_info_obj,
                dataset=dataset,
            )

            # append object
            ret_list.append(ret_obj)

        # return
        return ret_list

    def read_calibration(self,
                         file_list: Union[List[str], str],
                         n_parallel: int = 1,
                         quiet: bool = False,
                         dataset: Optional[Dataset] = None) -> List[Calibration]:
        """
        Read in UCalgary skymap files
        """
        # read data
        data = func_read_calibration(
            file_list,
            n_parallel=n_parallel,
            quiet=quiet,
        )

        # convert to return object
        ret_list = []
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
                        flat_field_multiplier_value = item[key]
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
            ret_obj = Calibration(
                filename=item_filename,
                version=version_str,
                dataset=dataset,
                detector_uid=detector_uid,
                flat_field_multiplier=flat_field_multiplier_value,
                rayleighs_perdn_persecond=rayleighs_perdn_persecond_value,
                generation_info=generation_info_obj,
            )

            # append object
            ret_list.append(ret_obj)

        # return
        return ret_list
