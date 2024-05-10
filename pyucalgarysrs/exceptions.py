"""
The exceptions module contains exceptions unique to the PyAuroraX library
"""


class SRSError(Exception):

    def __init__(self, *args, **kwargs):
        super(SRSError, self).__init__(*args, **kwargs)  # pragma: no cover


class SRSInitializationError(SRSError):
    """
    Error during library initialization occurred
    """
    pass


class SRSPurgeError(SRSError):
    """
    Error during purging of download or tar extraction working directory occurred
    """
    pass


class SRSAPIError(SRSError):
    """
    Error during API call occurred
    """
    pass


class SRSUnsupportedReadError(SRSError):
    """
    Unsupported dataset for read function
    """
    pass


class SRSDownloadError(SRSError):
    """
    Unsupported dataset for read function
    """
    pass
