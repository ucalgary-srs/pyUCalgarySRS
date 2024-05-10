import datetime
from typing import List, Optional, Union
from . import _list as module_list
from . import _download as module_download
from ._read import ReadManager
from ._schemas import *


class DataManager:

    # defaults
    __DEFAULT_DOWNLOAD_N_PARALLEL = 5

    def __init__(self, srs_obj):
        self.__srs_obj = srs_obj
        self.__list = module_list
        self.__download = module_download
        self.readers = ReadManager()

    def list_datasets(self, name: Optional[str] = None) -> List[Dataset]:
        """
        List available datasets
        """
        return self.__list.list_datasets(self.__srs_obj, name)

    def list_supported_read_datasets(self) -> List[str]:
        """
        List the datasets which have file reading capabilities supported.
        """
        return self.readers.list_supported_datasets()

    def check_if_read_supported(self, dataset_name: str) -> bool:
        """
        Check if a given dataset has file reading support
        """
        return self.readers.check_if_supported(dataset_name)

    def download(self,
                 dataset_name: str,
                 start: datetime.datetime,
                 end: datetime.datetime,
                 site_uid: Optional[str] = None,
                 n_parallel: int = __DEFAULT_DOWNLOAD_N_PARALLEL,
                 overwrite: bool = False,
                 progress_bar_disable: bool = False,
                 progress_bar_ncols: Optional[int] = None,
                 progress_bar_ascii: Optional[str] = None,
                 progress_bar_desc: Optional[str] = None,
                 timeout: Optional[int] = None) -> FileDownloadResult:
        """
        Download data from UCalgary open data archive
        """
        return self.__download._download_generic(
            self.__srs_obj,
            dataset_name,
            start,
            end,
            site_uid,
            n_parallel,
            overwrite,
            progress_bar_disable,
            progress_bar_ncols,
            progress_bar_ascii,
            progress_bar_desc,
            timeout,
        )

    def get_urls(self,
                 dataset_name: str,
                 start: datetime.datetime,
                 end: datetime.datetime,
                 site_uid: Optional[str] = None,
                 timeout: Optional[int] = None) -> FileListingResponse:
        """
        Get URLs of data files
        """
        return self.__download._get_urls(
            self.__srs_obj,
            dataset_name,
            start,
            end,
            site_uid,
            timeout,
        )

    def download_using_urls(self,
                            file_listing_response: FileListingResponse,
                            n_parallel: int = __DEFAULT_DOWNLOAD_N_PARALLEL,
                            overwrite: bool = False,
                            progress_bar_disable: bool = False,
                            progress_bar_ncols: Optional[int] = None,
                            progress_bar_ascii: Optional[str] = None,
                            progress_bar_desc: Optional[str] = None,
                            timeout: Optional[int] = None) -> FileDownloadResult:
        """
        Download data from UCalgary open data archive using a FileListingResponse 
        object. This would be used in cases where more customization is needed than
        the generic `download` function. For example, use `get_urls` to retrieve the
        list of URLs available for download, further process this list to fewer files
        based on some other requirement (ie. time downsampling such as one file per hour),
        and then use this function to download the custom set URLs.
        """
        return self.__download._download_using_urls(
            self.__srs_obj,
            file_listing_response,
            n_parallel,
            overwrite,
            progress_bar_disable,
            progress_bar_ncols,
            progress_bar_ascii,
            progress_bar_desc,
            timeout,
        )

    def read(self,
             dataset: Dataset,
             file_list: Union[List[str], str],
             n_parallel: int = 1,
             first_record: bool = False,
             no_metadata: bool = False,
             quiet: bool = False) -> Union[Data, List[Skymap], List[Calibration]]:
        """
        Read data files
        """
        return self.readers.read(
            dataset,
            file_list,
            n_parallel=n_parallel,
            first_record=first_record,
            no_metadata=no_metadata,
            quiet=quiet,
        )
