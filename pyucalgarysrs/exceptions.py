"""
The exceptions module contains exceptions unique to the PyAuroraX library
"""


class SRSException(Exception):

    def __init__(self, *args, **kwargs):
        super(SRSException, self).__init__(*args, **kwargs)  # pragma: no cover


class SRSInitializationException(SRSException):
    """
    Error during library initialization occurred
    """
    pass


class SRSPurgeException(SRSException):
    """
    Error during purging of download or tar extraction working directory occurred
    """
    pass


class SRSAPIException(SRSException):
    """
    Error during API call occurred
    """
    pass


class SRSUnsupportedReadException(SRSException):
    """
    Unsupported dataset for read function
    """
    pass


class SRSDownloadException(SRSException):
    """
    Unsupported dataset for read function
    """
    pass
