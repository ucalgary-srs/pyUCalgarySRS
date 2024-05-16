"""
The exceptions module contains unique exception classes utilized by PyUCalgarySRS. These 
exceptions can be used to help trap specific errors raised by the library.

Note that all exceptions are imported at the root level of the library. They
can be referenced using [`pyucalgarysrs.SRSError`](exceptions.html#pyucalgarysrs.exceptions.SRSError) 
or `pyucalgarysrs.exceptions.SRSError`.
"""


class SRSError(Exception):

    def __init__(self, *args, **kwargs):
        super(SRSError, self).__init__(*args, **kwargs)  # pragma: no cover


class SRSInitializationError(SRSError):
    """
    Error occurred during library initialization
    """
    pass


class SRSPurgeError(SRSError):
    """
    Error occurred during purging of download or tar extraction working directory
    """
    pass


class SRSAPIError(SRSError):
    """
    Error occurred during an API call
    """
    pass


class SRSUnsupportedReadError(SRSError):
    """
    Unsupported dataset for read function
    """
    pass


class SRSDownloadError(SRSError):
    """
    Error occurred during downloading of data
    """
    pass
