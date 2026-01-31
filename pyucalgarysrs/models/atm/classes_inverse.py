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
Classes for representing ATM inverse calculation requests and responses. All
classes in this module are included at the top level of this library.
"""

import datetime
from dataclasses import dataclass
from typing import Optional, Literal, Any
from numpy import ndarray


@dataclass
class ATMInverseOutputFlags:
    """
    Class to represent all output values included in an ATM inverse calculation.
    ATM calculations are performed in a way where you can toggle ON/OFF whichever
    pieces of information you do or don't want. This improves efficiency of the 
    calculation routine resulting in faster queries.

    By default, all output flags are set to False. There exist several helper methods
    to toggle all to True or toggle all to False. See the below Methods section for 
    details.

    Details about each output value can be found in the documentation for the 
    [`ATMInverseResult`](classes_inverse.html#pyucalgarysrs.models.atm.classes_inverse.ATMInverseResult)
    object.
    """
    energy_flux: bool = False
    mean_energy: bool = False
    oxygen_correction_factor: bool = False

    def set_all_true(self):
        """
        Sets all flags to `True`.
        """
        excluded_items = ["set_all_true", "set_all_false"]
        for var_name in dir(self):
            if (not var_name.startswith("__") and var_name not in excluded_items):
                setattr(self, var_name, True)

    def set_all_false(self):
        """
        Sets all flags to `False`.
        """
        excluded_items = ["set_all_true", "set_all_false"]
        for var_name in dir(self):
            if (not var_name.startswith("__") and var_name not in excluded_items):
                setattr(self, var_name, False)


@dataclass
class ATMInverseRequest:
    """
    Class that represents the UCalgary Space Remote Sensing API request when 
    performing an ATM inverse calculation. This object is included in any
    [`ATMInverseResultRequestInfo`](classes_inverse.html#pyucalgarysrs.models.atm.classes_inverse.ATMInverseResultRequestInfo) 
    and [`ATMInverseResult`](classes_inverse.html#pyucalgarysrs.models.atm.classes_inverse.ATMInverseResult) 
    objects.
    """
    timestamp: datetime.datetime
    geodetic_latitude: float
    geodetic_longitude: float
    intensity_4278: float
    intensity_5577: float
    intensity_6300: float
    intensity_8446: float
    precipitation_flux_spectral_type: Literal["gaussian", "maxwellian"]
    nrlmsis_model_version: Literal["00", "2.0"]
    atm_model_version: Literal["1.0"]
    special_logic_keyword: Optional[str]
    output: ATMInverseOutputFlags
    no_cache: bool


@dataclass
class ATMInverseResultRequestInfo:
    """
    Class containing details about interacting with the UCalgary Space Remote Sensing
    API when performing an ATM inverse calculation. It contains the API request itself,
    and other information returned by the API.

    An instance of this is included in any
    [`ATMInverseResult`](classes_inverse.html#pyucalgarysrs.models.atm.classes_inverse.ATMInverseResult) 
    object.

    Attributes:
        request (ATMInverseRequest): 
            Instance of class that represents the lower-level API request when performing 
            an ATM inverse calculation.

        calculation_duration_ms (float): 
            Duration the the API spent performing the ATM inverse calculation. Represented
            in milliseconds.
    """
    request: ATMInverseRequest
    calculation_duration_ms: float


@dataclass
class ATMInverseResult:
    """
    Class containing all data from an ATM inverse calculation. This class also includes
    details about the input parameters to inversion calculation routine (`request_info`),
    along with all output data values.
    
    Based on the request's output flags, if flags were set to False then the values will 
    be None types. If flags were set to True then the values will be their respective 
    types (e.g., float, numpy ndarray).

    Attributes:
        request_info (ATMInverseResultRequestInfo): 
            Information about the API request made to perform the ATM forward calculation.
        
        energy_flux (float): 
            Derived energy flux in erg/cm2/s.

        mean_energy (float): 
            Derived characteristic energy in EV. Previously named as 'characteristic_energy'.

        oxygen_correction_factor (float): 
            Derived oxygen correction factor.
    """
    request_info: ATMInverseResultRequestInfo
    energy_flux: Any
    mean_energy: Any
    oxygen_correction_factor: Any

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "ATMInverseResult(energy_flux=%f, mean_energy=%f, oxygen_correction_factor=%f, ...)" % (
            self.energy_flux,
            self.mean_energy,
            self.oxygen_correction_factor,
        )

    def pretty_print(self):
        """
        A special print output for this class.
        """
        print("ATMInverseResult:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
                continue

            # convert var to string format we want
            var_value = getattr(self, var_name)
            var_str = "None"
            if (var_name == "request_info"):
                var_str = "ATMInverseResultRequestInfo(...)"
            if (var_value is not None):
                if (isinstance(var_value, float)):
                    var_str = "%f" % (var_value)
                elif (isinstance(var_value, ndarray)):
                    var_str = "%s ...])" % (var_value.__repr__()[0:60])  # pragma: nocover-ok

            # print string for this var
            print("  %-26s: %s" % (var_name, var_str))
