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
__version__ = "0.0.27"

# documentation
__pdoc__ = {"pyucalgarysrs": False}
__all__ = ["PyUCalgarySRS"]

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
