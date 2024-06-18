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
Classes for representing data download and reading operations. All
classes in this module are included at the top level of this library.
"""

import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal, Any
from numpy import ndarray


class Dataset:
    """
    A dataset available from the UCalgary Space Remote Sensing API, with possibly
    support for downloading and/or reading.

    Attributes:
        name (str): 
            Dataset name
        
        short_description (str): 
            A short description about the dataset
        
        long_description (str): 
            A longer description about the dataset
        
        data_tree_url (str): 
            The data tree URL prefix. Used for saving data locally with a similar data tree 
            structure compared to the UCalgary Open Data archive.
        
        file_listing_supported (bool): 
            Flag indicating if file listing (downloading) is supported for this dataset.
        
        file_reading_supported (bool): 
            Flag indicating if file reading is supported for this dataset.
        
        level (str): 
            Dataset level as per L0/L1/L2/etc standards.
        
        doi (str): 
            Dataset DOI unique identifier.
        
        doi_details (str): 
            Further details about the DOI.
        
        citation (str): 
            String to use when citing usage of the dataset.
        
        provider (str): 
            Data provider.
    """

    def __init__(self,
                 name: str,
                 short_description: str,
                 long_description: str,
                 data_tree_url: str,
                 file_listing_supported: bool,
                 file_reading_supported: bool,
                 level: str,
                 doi: Optional[str] = None,
                 doi_details: Optional[str] = None,
                 citation: Optional[str] = None):
        self.name = name
        self.short_description = short_description
        self.long_description = long_description
        self.data_tree_url = data_tree_url
        self.file_listing_supported = file_listing_supported
        self.file_reading_supported = file_reading_supported
        self.level = level
        self.doi = doi
        self.doi_details = doi_details
        self.citation = citation
        self.provider = "UCalgary"

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "Dataset(name=%s, short_description='%s', provider='%s', level='%s', doi_details='%s', ...)" % (
            self.name,
            self.short_description,
            self.provider,
            self.level,
            self.doi_details,
        )

    def pretty_print(self):
        """
        A special print output for this class.
        """
        print("Dataset:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
                continue

            # convert var to string format we want
            var_value = getattr(self, var_name)
            print("  %-27s: %s" % (var_name, None if var_value is None else var_value))


@dataclass
class FileListingResponse:
    """
    Representation of the file listing response from the UCalgary Space Remote Sensing API.

    Attributes:
        urls (List[str]): 
            A list of URLs for available data files.
        
        path_prefix (str): 
            The URL prefix, which is sed for saving data locally with a similar data tree 
            structure compared to the UCalgary Open Data archive.
        
        count (int): 
            The number of URLs available.
        
        dataset (Dataset): 
            The `Dataset` object for this data.
        
        total_bytes (int): 
            The cumulative amount of bytes for the available URLs.
    """
    urls: List[str]
    path_prefix: str
    count: int
    dataset: Dataset
    total_bytes: Optional[int] = None


@dataclass
class FileDownloadResult:
    """
    Representation of the results from a data download call.

    Attributes:
        filenames (List[str]): 
            List of downloaded files, as absolute paths of their location on the local machine.
        
        count (int): 
            Number of files downloaded
        
        total_bytes (int): 
            Cumulative amount of bytes saved on the local machine.
        
        output_root_path (str): 
            The root path of where the data was saved to on the local machine.
        
        dataset (Dataset): 
            The `Dataset` object for this data.
    """
    filenames: List[str]
    count: int
    total_bytes: int
    output_root_path: str
    dataset: Dataset


@dataclass
class ProblematicFile:
    """
    Representation about a file that had issues being read.

    Attributes:
        filename (str): 
            Filename of the problematic file.
        
        error_message (str): 
            Error message that was encountered while attempting to read the file.
        
        error_type (str): 
            Error type encountered. Possible values are 'error' or 'warning'.
    """
    filename: str
    error_message: str
    error_type: Literal["error", "warning"]


@dataclass
class SkymapGenerationInfo:
    """
    Representation of generation details for a specific skymap file.

    Attributes:
        author (str): 
            Name of individual who created the skymap
        
        ccd_center (float): 
            Center pixels of the CCD
        
        code_used (str): 
            Program name for the code used to generate the skymap
        
        data_loc (str): 
            Location of the data on the UCalgary data systems used during generation
        
        date_generated (datetime.datetime): 
            Timestamp of when the skymap was generated
        
        date_time_used (datetime.datetime): 
            Timestamp of what hour was used during generation
        
        img_flip (ndarray): 
            Image flipping specifics
        
        optical_orientation (ndarray): 
            Image orientation details
        
        optical_projection (ndarray): 
            Image projection details
        
        pixel_aspect_ratio (float): 
            Aspect ratio details
        
        valid_interval_start (datetime.datetime): 
            Valid start time for this skymap
        
        valid_interval_end (datetime.datetime): 
            Valid end time for this skymap. If None, then end time is unbounded and valid up until 
            the next newest skymap.
    """
    author: str
    ccd_center: float
    code_used: str
    data_loc: str
    date_generated: datetime.datetime
    date_time_used: datetime.datetime
    img_flip: ndarray
    optical_orientation: ndarray
    optical_projection: ndarray
    pixel_aspect_ratio: float
    valid_interval_start: datetime.datetime
    valid_interval_stop: Optional[datetime.datetime] = None
    bytscl_values: Optional[ndarray] = None

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "SkymapGenerationInfo(date_generated=%s, author='%s', ccd_center=%s, ...)" % (
            str(self.date_generated.__repr__()),
            self.author,
            self.ccd_center,
        )

    def pretty_print(self):
        """
        A special print output for this class.
        """
        print("SkymapGenerationInfo:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
                continue

            # convert var to string format we want
            var_value = getattr(self, var_name)
            var_str = "None"
            if (var_value is not None):
                if (isinstance(var_value, ndarray)):
                    var_str = "array(dims=%s, dtype=%s)" % (var_value.shape, var_value.dtype)
                else:
                    var_str = str(var_value)

            # print string for this var
            print("  %-24s: %s" % (var_name, var_str))


@dataclass
class Skymap:
    """
    Representation for a skymap file.

    Attributes:
        filename (str): 
            Filename for the skymap file, as an absolute path of its location on the local machine.
        
        project_uid (str): 
            Project unique identifier
        
        site_uid (str): 
            Site unique identifier
        
        imager_uid (str): 
            Imager/device unique identifier
        
        site_map_latitude (float): 
            Geodetic latitude of instrument
        
        site_map_longitude (float): 
            Geodetic longitude of instrument
        
        site_map_altitude (float): 
            Altitude of the instrument (in meters)
        
        full_elevation (ndarray): 
            Elevation angle from horizon, for each image pixel (in degrees)
        
        full_azimuth (ndarray): 
            Local azimuth angle from 0 degrees north, positive moving east (in degrees)
        
        full_map_altitude (ndarray): 
            Altitudes that image coordinates are mapped to (in kilometers)
        
        full_map_latitude (ndarray): 
            Geodetic latitudes of pixel corners, mapped to various altitudes (specified by `full_map_altitude`)
        
        full_map_longitude (ndarray): 
            Geodetic longitudes of pixel corners, mapped to various altitudes (specified by `full_map_altitude`)
        
        generation_info (SkymapGenerationInfo): 
            Metadata describing details about this skymap's generation process
        
        version (str): 
            Version of the skymap
        
        dataset (Dataset): 
            The `Dataset` object for this data.
    """
    filename: str
    project_uid: str
    site_uid: str
    imager_uid: str
    site_map_latitude: float
    site_map_longitude: float
    site_map_altitude: float
    full_elevation: ndarray
    full_azimuth: ndarray
    full_map_altitude: ndarray
    full_map_latitude: ndarray
    full_map_longitude: ndarray
    generation_info: SkymapGenerationInfo
    version: str

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "Skymap(project_uid=%s, site_uid=%s, imager_uid=%s, site_map_latitude=%f, site_map_longitude=%f, ...)" % (
            self.project_uid,
            self.site_uid,
            self.imager_uid,
            self.site_map_latitude,
            self.site_map_longitude,
        )

    def pretty_print(self):
        """
        A special print output for this class.
        """
        print("Skymap:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
                continue

            # convert var to string format we want
            var_value = getattr(self, var_name)
            var_str = "None"
            if (var_name == "generation_info"):
                var_str = "SkymapGenerationInfo(...)"
            elif (var_value is not None):
                if (isinstance(var_value, ndarray)):
                    var_str = "array(dims=%s, dtype=%s)" % (var_value.shape, var_value.dtype)
                else:
                    var_str = str(var_value)

            # print string for this var
            print("  %-23s: %s" % (var_name, var_str))

    def get_precalculated_altitudes(self):
        """
        Get the altitudes that have been precalculated in this skymap. Units are kilometers.
        """
        alts_km = [float(x / 1000.) for x in self.full_map_altitude]
        return alts_km


@dataclass
class CalibrationGenerationInfo:
    """
    Representation of generation details for a specific calibration file.

    Attributes:
        valid_interval_start (datetime.datetime): 
            Valid start timestamp for this calibration file
        
        valid_interval_end (datetime.datetime): 
            Valid end time for this calibration file. If None, then end time is unbounded and valid up 
            until the next newest calibration for this detector UID.
        
        author (str): 
            Individual who generated the calibration file
        
        input_data_dir (str): 
            Path on UCalgary data system to the raw calibration files
        
        skymap_filename (str): 
            Path to skymap file used to assist with calibration process. If None, no skymap file was used.
    """
    valid_interval_start: datetime.datetime
    valid_interval_stop: Optional[datetime.datetime] = None
    author: Optional[str] = None
    input_data_dir: Optional[str] = None
    skymap_filename: Optional[str] = None

    def pretty_print(self):
        """
        A special print output for this class.
        """
        print("CalibrationGenerationInfo:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
                continue

            # convert var to string format we want
            var_value = getattr(self, var_name)
            var_str = "None" if var_value is None else str(var_value)

            # print string for this var
            print("  %-25s: %s" % (var_name, var_str))


@dataclass
class Calibration:
    """
    Representation for a calibration file.

    Attributes:
        filename (str): 
            Filename for the calibration file, as an absolute path of its location on the local machine.
        
        detector_uid (str): 
            Detector/imager/camera unique identifier
        
        version (str): 
            Version number of the calibration file
        
        generation_info (CalibrationGenerationInfo): 
            Metadata describing details about this calibration's generation process
        
        rayleighs_perdn_persecond (float): 
            Calibrated value for Rayleighs per data number per second (R/dn/s). This value will be None 
            if a flatfield calibration file was read instead of a rayleighs calibration file.
        
        flat_field_multiplier (ndarray): 
            Calibrated flat field array. This value will be None if a rayleighs calibration file was 
            read instead of a flatfield calibration file.
        
        dataset (Dataset): 
            The `Dataset` object for this data.
    """
    filename: str
    detector_uid: str
    version: str
    generation_info: CalibrationGenerationInfo
    rayleighs_perdn_persecond: Optional[float] = None
    flat_field_multiplier: Optional[ndarray] = None
    dataset: Optional[Dataset] = None

    def pretty_print(self):
        """
        A special print output for this class.
        """
        print("Calibration:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
                continue

            # convert var to string format we want
            var_value = getattr(self, var_name)
            var_str = "None"
            if (var_name == "generation_info"):
                var_str = "CalibrationGenerationInfo(...)"
            elif (var_name == "dataset" and var_value is not None):
                var_str = "Dataset(...)"
            elif (var_value is not None):
                if (isinstance(var_value, ndarray)):
                    var_str = "array(dims=%s, dtype=%s)" % (var_value.shape, var_value.dtype)
                else:
                    var_str = str(var_value)

            # print string for this var
            print("  %-30s: %s" % (var_name, var_str))


@dataclass
class Data:
    """
    Representation of the data read in from a `read` call.

    Attributes:
        data (Any): 
            The loaded data. This can be one of the following types: ndarray, List[Skymap], List[Calibration].
        
        timestamp (List[datetime.datetime]): 
            List of timestamps for the read in data.
        
        metadata (List[Dict]): 
            List of dictionaries containing metadata specific to each timestamp/image/record.
        
        problematic_files (List[ProblematicFiles]): 
            A list detailing any files that encountered issues during reading.
        
        calibrated_data (Any): 
            A calibrated version of the data. Populated and utilized by data analysis libraries. Has a `None` value
            until calibrated data is inserted manually.

        dataset (Dataset): 
            The `Dataset` object for this data.
    """
    data: Any
    timestamp: List[datetime.datetime]
    metadata: List[Dict]
    problematic_files: List[ProblematicFile]
    calibrated_data: Any
    dataset: Optional[Dataset] = None

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        # set data value
        if (isinstance(self.data, ndarray) is True):
            data_str = "array(dims=%s, dtype=%s)" % (self.data.shape, self.data.dtype)
        elif (isinstance(self.data, list) is True):
            if (len(self.data) == 0):
                data_str = "[0 items]"
            elif (isinstance(self.data[0], Skymap) is True):
                if (len(self.data) == 1):
                    data_str = "[1 Skymap object]"
                else:
                    data_str = "[%d Skymap objects]" % (len(self.data))
            elif (isinstance(self.data[0], Calibration) is True):
                if (len(self.data) == 1):
                    data_str = "[1 Calibration object]"
                else:
                    data_str = "[%d Calibration objects]" % (len(self.data))
            else:
                data_str = "[%d items]" % (len(self.data))
        else:
            data_str = self.data.__repr__()

        # set timestamp string
        if (len(self.timestamp) == 0):
            timestamp_str = "[]"
        elif (len(self.timestamp) == 1):
            timestamp_str = "[1 datetime]"
        else:
            timestamp_str = "[%d datetimes]" % (len(self.timestamp))

        # set metadata string
        if (len(self.metadata) == 0):
            metadata_str = "[]"
        elif (len(self.metadata) == 1):
            metadata_str = "[1 dictionary]"
        else:
            metadata_str = "[%d dictionaries]" % (len(self.timestamp))

        # set rest of values
        problematic_files_str = "[]" if len(self.problematic_files) == 0 else "[%d problematic files]" % (len(self.problematic_files))
        calibrated_data_str = "None" if self.calibrated_data is None else "array(dims=%s, dtype=%s)" % (self.calibrated_data.shape,
                                                                                                        self.calibrated_data.dtype)
        dataset_str = "None" if self.dataset is None else self.dataset.__repr__()[0:75] + "...)"

        # return
        return "Data(data=%s, timestamp=%s, metadata=%s, problematic_files=%s, calibrated_data=%s, dataset=%s)" % (
            data_str,
            timestamp_str,
            metadata_str,
            problematic_files_str,
            calibrated_data_str,
            dataset_str,
        )

    def pretty_print(self):
        """
        A special print output for this class.
        """
        # set data value
        if (isinstance(self.data, ndarray) is True):
            data_str = "array(dims=%s, dtype=%s)" % (self.data.shape, self.data.dtype)
        elif (isinstance(self.data, list) is True):
            if (len(self.data) == 0):
                data_str = "[0 items]"
            elif (isinstance(self.data[0], Skymap) is True):
                if (len(self.data) == 1):
                    data_str = "[1 Skymap object]"
                else:
                    data_str = "[%d Skymap objects]" % (len(self.data))
            elif (isinstance(self.data[0], Calibration) is True):
                if (len(self.data) == 1):
                    data_str = "[1 Calibration object]"
                else:
                    data_str = "[%d Calibration objects]" % (len(self.data))
            else:
                data_str = "[%d items]" % (len(self.data))
        else:
            data_str = self.data.__repr__()

        # set timestamp string
        if (len(self.timestamp) == 0):
            timestamp_str = "[]"
        elif (len(self.timestamp) == 1):
            timestamp_str = "[1 datetime]"
        else:
            timestamp_str = "[%d datetimes]" % (len(self.timestamp))

        # set metadata string
        if (len(self.metadata) == 0):
            metadata_str = "[]"
        elif (len(self.metadata) == 1):
            metadata_str = "[1 dictionary]"
        else:
            metadata_str = "[%d dictionaries]" % (len(self.timestamp))

        # set rest of values
        problematic_files_str = "[]" if len(self.problematic_files) == 0 else "[%d problematic files]" % (len(self.problematic_files))
        calibrated_data_str = "None" if self.calibrated_data is None else "array(dims=%s, dtype=%s)" % (self.calibrated_data.shape,
                                                                                                        self.calibrated_data.dtype)
        dataset_str = "None" if self.dataset is None else self.dataset.__repr__()[0:75] + "...)"

        # print
        print("Data:")
        print("  %-22s: %s" % ("data", data_str))
        print("  %-22s: %s" % ("timestamp", timestamp_str))
        print("  %-22s: %s" % ("metadata", metadata_str))
        print("  %-22s: %s" % ("problematic_files", problematic_files_str))
        print("  %-22s: %s" % ("calibrated_data", calibrated_data_str))
        print("  %-22s: %s" % ("dataset", dataset_str))


class Observatory:
    """
    Representation for an observatory.

    Attributes:
        uid (str): 
            4-letter unique identifier (traditionally referred to as the site UID)

        full_name (str): 
            full location string for the observatory
        
        geodetic_latitude (float): 
            geodetic latitude for the observatory, in decimal format (-90 to 90)
        
        geodetic_longitude (float): 
            geodetic longitude for the observatory, in decimal format (-180 to 180)

        provider (str): 
            Data provider.
    """

    def __init__(self, uid: str, full_name: str, geodetic_latitude: float, geodetic_longitude: float):
        self.uid = uid
        self.full_name = full_name
        self.geodetic_latitude = geodetic_latitude
        self.geodetic_longitude = geodetic_longitude
        self.provider = "UCalgary"

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "Observatory(uid=%s, full_name='%s', geodetic_latitude=%s, geodetic_longitude=%s, provider='%s')" % (
            self.uid,
            self.full_name,
            self.geodetic_latitude,
            self.geodetic_longitude,
            self.provider,
        )

    def pretty_print(self):
        """
        A special print output for this class.
        """
        print("Observatory:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
                continue

            # convert var to string format we want
            var_value = getattr(self, var_name)
            print("  %-22s: %s" % (var_name, None if var_value is None else var_value))
