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
