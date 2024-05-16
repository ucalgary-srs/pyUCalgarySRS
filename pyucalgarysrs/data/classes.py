"""
Classes for representing data download and reading operations. All
classes in this module are included at the top level of this library.
"""

import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal
from numpy import ndarray


@dataclass
class Dataset:
    """
    A dataset available from the UCalgary Space Remote Sensing API, with possibly
    support for downloading and/or reading.

    Attributes:
        name (str): Dataset name
        short_description (str): A short description about the dataset
        long_description (str): A longer description about the dataset
        data_tree_url (str): The data tree URL prefix. Used for saving data locally with a 
            similar data tree structure compared to the UCalgary Open Data archive.
        file_listing_supported (bool): Flag indicating if file listing (downloading) is 
            supported for this dataset.
        file_reading_supported (bool): Flag indicating if file reading is supported for this
            dataset.
        level (str): Dataset level as per L0/L1/L2/etc standards.
        doi (str): Dataset DOI unique identifier.
        doi_details (str): Further details about the DOI.
        citation (str): String to use when citing usage of the dataset.
    """
    name: str
    short_description: str
    long_description: str
    data_tree_url: str
    file_listing_supported: bool
    file_reading_supported: bool
    level: str
    doi: Optional[str] = None
    doi_details: Optional[str] = None
    citation: Optional[str] = None

    def show_acknowledgement_info(self):
        """
        A special print output for the dataset's acknowledgement information.
        """
        print("\n%-15s%s" % ("DOI:", self.doi))
        print("%-15s%s" % ("DOI details:", self.doi_details))
        print("%-15s\"%s\"\n" % ("Citation:", self.citation))


@dataclass
class FileListingResponse:
    """
    Representation of the file listing response from the UCalgary Space Remote Sensing API.

    Attributes:
        urls (List[str]): A list of URLs for available data files.
        path_prefix (str): The URL prefix, which is sed for saving data locally with a 
            similar data tree structure compared to the UCalgary Open Data archive.
        count (int): The number of URLs available.
        dataset (Dataset): The `Dataset` object for this data.
        total_bytes (int): The cumulative amount of bytes for the available URLs.
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
        filenames (List[str]): List of downloaded files, as absolute paths of their location on 
            the local machine.
        count (int): Number of files downloaded
        total_bytes (int): Cumulative amount of bytes saved on the local machine.
        output_root_path (str): The root path of where the data was saved to on the local machine.
        dataset (Dataset): The `Dataset` object for this data.
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
        filename (str): Filename of the problematic file.
        error_message (str): Error message that was encountered while attempting to read the file.
        error_type (str): Error type encountered. Possible values are 'error' or 'warning'.
    """
    filename: str
    error_message: str
    error_type: Literal["error", "warning"]


@dataclass
class Data:
    """
    Representation of the data read in from a `read` call.

    Attributes:
        data (ndarray): Numpy n-dimensional array containing the data read in.
        timestamp (List[datetime.datetime]): List of timestamps for the read in data.
        metadata (List[Dict]): List of dictionaries containing metadata specific to each
            timestamp/image/record.
        problematic_files (List[ProblematicFiles]): A list detailing any files that encountered
            issues during reading.
        dataset (Dataset): The `Dataset` object for this data.
    """
    data: ndarray
    timestamp: List[datetime.datetime]
    metadata: List[Dict]
    problematic_files: List[ProblematicFile]
    dataset: Optional[Dataset] = None

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        data_str = "array(dims=%s, dtype=%s)" % (self.data.shape, self.data.dtype)
        timestamp_str = "[%d datetime objects]" % (len(self.timestamp))
        metadata_str = "[%d dictionaries]" % (len(self.metadata))
        problematic_files_str = self.problematic_files.__repr__()
        dataset_str = "unknown" if self.dataset is None else self.dataset.__repr__()[0:75] + "...)"

        return "Data(data=%s, timestamp=%s, metadata=%s, problematic_files=%s, dataset=%s)" % (
            data_str,
            timestamp_str,
            metadata_str,
            problematic_files_str,
            dataset_str,
        )


@dataclass
class SkymapGenerationInfo:
    """
    Representation of generation details for a specific skymap file.

    Attributes:
        author (str): Name of individual who created the skymap
        ccd_center (float): Center pixels of the CCD
        code_used (str): Program name for the code used to generate the skymap
        data_loc (str): Location of the data on the UCalgary data systems used during generation
        date_generated (datetime.datetime): Timestamp of when the skymap was generated
        date_time_used (datetime.datetime): 
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
        return "SkymapGenerationInfo(date_generated=%s, author=%s, ccd_center=%s, ...)" % (
            str(self.date_generated.__repr__()),
            self.author,
            self.ccd_center,
        )


@dataclass
class Skymap:
    filename: str
    project_uid: str
    site_uid: str
    imager_uid: str
    site_map_latitude: float
    site_map_longitude: float
    site_map_altitude: float
    bin_row: ndarray
    bin_column: ndarray
    bin_elevation: ndarray
    bin_azimuth: ndarray
    bin_map_altitude: float
    bin_map_latitude: ndarray
    bin_map_longitude: ndarray
    full_row: ndarray
    full_column: ndarray
    full_ignore: ndarray
    full_subtract: ndarray
    full_multiply: ndarray
    full_elevation: ndarray
    full_azimuth: ndarray
    full_map_altitude: ndarray
    full_map_latitude: ndarray
    full_map_longitude: ndarray
    full_bin: ndarray
    generation_info: SkymapGenerationInfo
    version: str
    dataset: Optional[Dataset] = None

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        dataset_str = "unknown" if self.dataset is None else self.dataset.__repr__()[0:75] + "...)"
        return "Skymap(project_uid=%s, site_uid=%s, imager_uid=%s, site_map_latitude=%f, site_map_longitude=%f, dataset=%s, ...)" % (
            self.project_uid,
            self.site_uid,
            self.imager_uid,
            self.site_map_latitude,
            self.site_map_longitude,
            dataset_str,
        )


@dataclass
class CalibrationGenerationInfo:
    valid_interval_start: datetime.datetime
    author: Optional[str] = None
    valid_interval_stop: Optional[datetime.datetime] = None
    input_data_dir: Optional[str] = None
    skymap_filename: Optional[str] = None


@dataclass
class Calibration:
    filename: str
    detector_uid: str
    version: str
    generation_info: CalibrationGenerationInfo
    rayleighs_perdn_persecond: Optional[float] = None
    flat_field_multiplier: Optional[ndarray] = None
    dataset: Optional[Dataset] = None
