import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal
from numpy import ndarray


@dataclass
class Dataset:
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


@dataclass
class FileListingResponse:
    urls: List[str]
    path_prefix: str
    count: int
    dataset: Dataset
    total_bytes: Optional[int] = None


@dataclass
class FileDownloadResult:
    filenames: List[str]
    count: int
    total_bytes: int
    output_root_path: str
    dataset: Dataset


@dataclass
class ProblematicFile:
    filename: str
    error_message: str
    error_type: Literal["error", "warning"]


@dataclass
class Data:
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
