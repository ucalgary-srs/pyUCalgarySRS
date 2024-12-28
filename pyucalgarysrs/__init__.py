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
The PyUCalgarySRS library provides a way to interact with the UCalgary Space
Remote Sensing (SRS) Open Data Platform. It enables users to programmatically
download and read data, and utilize the TREx Auroral Transport Model (ATM). 
This library leverages the UCalgary SRS API for some functions.

For more information about the data available and usage examples, visit the 
[Open Data Platform website](https://data.phys.ucalgary.ca).

Installation:
```console
$ pip install pyucalgarysrs
```

Basic usage:
```python
> import pyucalgarysrs
> srs = pyucalgarysrs.PyUCalgarySRS()
```
"""

# versioning info
__version__ = "1.7.0"

# documentation
__pdoc__ = {"pyucalgarysrs": False}
__all__ = [
    "PyUCalgarySRS",
    "__version__",
]

# pull in top level class
from .pyucalgarysrs import PyUCalgarySRS

# pull in exceptions
from .exceptions import (
    SRSError,
    SRSInitializationError,
    SRSPurgeError,
    SRSAPIError,
    SRSUnsupportedReadError,
    SRSDownloadError,
)

# pull in schemas
from .data.classes import (
    Dataset,
    FileListingResponse,
    FileDownloadResult,
    Calibration,
    CalibrationGenerationInfo,
    Data,
    Skymap,
    SkymapGenerationInfo,
    ProblematicFile,
    Observatory,
)
from .models.atm.classes_forward import (
    ATMForwardOutputFlags,
    ATMForwardRequest,
    ATMForwardResult,
    ATMForwardResultRequestInfo,
)
from .models.atm.classes_inverse import (
    ATMInverseForwardParams,
    ATMInverseOutputFlags,
    ATMInverseRequest,
    ATMInverseResult,
    ATMInverseResultRequestInfo,
)
