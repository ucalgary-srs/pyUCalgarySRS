import os
import shutil
from typing import Optional, Union, Dict
from pathlib import Path
from .exceptions import SRSInitializationError, SRSPurgeError
from .data import DataManager
from .models import ModelsManager
from . import __version__


class PyUCalgarySRS:
    """
    Top-level class for interacting with UCalgary SRS data tools
    """
    DEFAULT_API_BASE_URL = "https://api.phys.ucalgary.ca"
    DEFAULT_API_TIMEOUT = 10
    DEFAULT_API_HEADERS = {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": "python-pyaurorax/%s" % (__version__),
    }

    def __init__(self,
                 download_output_root_path: Optional[str] = None,
                 read_tar_temp_dir: Optional[str] = None,
                 api_base_url: str = DEFAULT_API_BASE_URL,
                 api_key: Optional[Union[str, None]] = None,
                 api_timeout: int = DEFAULT_API_TIMEOUT,
                 api_headers: Optional[Dict] = DEFAULT_API_HEADERS):
        # public parameters
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.api_headers = api_headers
        self.api_timeout = api_timeout
        self.in_jupyter_notebook = self.__initialize_jupyter_flag()

        # private parameters exposed publicly using decorators
        self.__download_output_root_path = download_output_root_path
        self.__read_tar_temp_dir = read_tar_temp_dir

        # initialize paths
        self.initialize_paths()

        # initialize sub-modules
        self.data = DataManager(self)
        self.models = ModelsManager(self)

    # special methods
    # -----------------------------
    def __str__(self) -> str:
        """
        String method

        Returns:
            string format of PyUCalgarySRS object
        """
        return self.__repr__()

    def __repr__(self) -> str:
        """
        Object representation

        Returns:
            PyUCalgarySRS object representation
        """
        return ("PyUCalgarySRS(download_output_root_path='%s', read_tar_temp_dir='%s', api_base_url='%s', " +
                "api_headers=%s, api_timeout=%s, in_jupyter_notebook=%s)") % (
                    self.__download_output_root_path,
                    self.__read_tar_temp_dir,
                    self.api_base_url,
                    self.api_headers,
                    self.api_timeout,
                    self.in_jupyter_notebook,
                )

    def __initialize_jupyter_flag(self):
        """
        Check if class has been instantiated within a Jupyter notebook. If so, we will use
        tqdm progress bars for notebooks specifically, among other future things.
        """
        try:
            from IPython import get_ipython  # type: ignore
            shell = get_ipython().__class__.__name__
            if (shell == "ZMQInteractiveShell"):  # pragma: nocover
                return True  # jupyter notebook or qtconsole
            elif (shell == "TerminalInteractiveShell"):  # pragma: nocover
                return False  # terminal running IPython
            else:
                return False  # other unknown type
        except (ImportError, NameError):  # pragma: nocover
            return False  # probably standard Python interpreter

    # -----------------------------
    # properties
    # -----------------------------
    @property
    def download_output_root_path(self):
        return str(self.__download_output_root_path)

    @download_output_root_path.setter
    def download_output_root_path(self, value: str):
        self.__download_output_root_path = value
        self.initialize_paths()

    @property
    def read_tar_temp_dir(self):
        return str(self.__read_tar_temp_dir)

    @read_tar_temp_dir.setter
    def read_tar_temp_dir(self, value: str):
        self.__read_tar_temp_dir = value
        self.initialize_paths()

    # -----------------------------
    # public methods
    # -----------------------------
    def purge_download_output_root_path(self):
        try:
            for item in os.listdir(self.download_output_root_path):
                item = "%s/%s" % (self.download_output_root_path, item)
                if (os.path.isdir(item) is True):
                    shutil.rmtree(item)
                elif (os.path.isfile(item) is True):
                    os.remove(item)
        except Exception as e:  # pragma: nocover
            raise SRSPurgeError("Error while purging download output root path: %s" % (str(e))) from e

    def purge_read_tar_temp_dir(self):
        try:
            for item in os.listdir(self.read_tar_temp_dir):
                item = "%s/%s" % (self.read_tar_temp_dir, item)
                if (os.path.isdir(item) is True):
                    shutil.rmtree(item)
                elif (os.path.isfile(item) is True):
                    os.remove(item)
        except Exception as e:  # pragma: nocover
            raise SRSPurgeError("Error while purging read tar temp dir: %s" % (str(e))) from e

    def initialize_paths(self):
        if (self.__download_output_root_path is None):
            self.__download_output_root_path = Path("%s/pyucalgarysrs_data" % (str(Path.home())))
        if (self.__read_tar_temp_dir is None):
            self.__read_tar_temp_dir = Path("%s/.tar_temp_working" % (self.__download_output_root_path))
        try:
            os.makedirs(self.download_output_root_path, exist_ok=True)
            os.makedirs(self.read_tar_temp_dir, exist_ok=True)
        except IOError as e:  # pragma: nocover
            raise SRSInitializationError("Error during output path creation: %s" % str(e)) from e
