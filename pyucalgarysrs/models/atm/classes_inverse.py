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
    characteristic_energy: bool = False
    oxygen_correction_factor: bool = False
    height_integrated_rayleighs_4278: bool = False
    height_integrated_rayleighs_5577: bool = False
    height_integrated_rayleighs_6300: bool = False
    height_integrated_rayleighs_8446: bool = False
    altitudes: bool = False
    emission_4278: bool = False
    emission_5577: bool = False
    emission_6300: bool = False
    emission_8446: bool = False
    plasma_electron_density: bool = False
    plasma_o2plus_density: bool = False
    plasma_oplus_density: bool = False
    plasma_noplus_density: bool = False
    plasma_ionisation_rate: bool = False
    plasma_electron_temperature: bool = False
    plasma_ion_temperature: bool = False
    plasma_pederson_conductivity: bool = False
    plasma_hall_conductivity: bool = False
    neutral_o2_density: bool = False
    neutral_o_density: bool = False
    neutral_n2_density: bool = False
    neutral_n_density: bool = False
    neutral_temperature: bool = False

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
    atm_model_version: Literal["1.0"]
    timestamp: datetime.datetime
    geodetic_latitude: float
    geodetic_longitude: float
    intensity_4278: float
    intensity_5577: float
    intensity_6300: float
    intensity_8446: float
    precipitation_flux_spectral_type: Literal["gaussian", "maxwellian"]
    nrlmsis_model_version: Literal["00", "2.0"]
    atmospheric_attenuation_correction: bool
    output: ATMInverseOutputFlags
    no_cache: bool


@dataclass
class ATMInverseForwardParams:
    """
    Class representing a forward calculation done under-the-hood of an inverse
    calculation.

    Depending on the inversion request parameters, a further forward calculation 
    may be performed by the API. This variable contains the details of any such 
    calculation.
    """
    maxwellian_energy_flux: float
    gaussian_energy_flux: float
    maxwellian_characteristic_energy: float
    gaussian_peak_energy: float
    gaussian_spectral_width: float
    nrlmsis_model_version: Literal["00", "2.0"]
    oxygen_correction_factor: float
    timescale_auroral: int
    timescale_transport: int


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
        
        forward_params (ATMInverseForwardParams): 
            Depending on the inversion request parameters, a further forward calculation may be performed 
            under-the-hood. This variable contains the details of any such calculation.
    """
    request: ATMInverseRequest
    calculation_duration_ms: float
    forward_params: Optional[ATMInverseForwardParams] = None


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

        characteristic_energy (float): 
            Derived characteristic energy in EV.

        oxygen_correction_factor (float): 
            Derived oxygen correction factor.

        height_integrated_rayleighs_4278 (float): 
            Height-integrated Rayleighs value for the 427.8nm emission (blue).

        height_integrated_rayleighs_5577 (float): 
            Height-integrated Rayleighs value for the 557.7nm emission (green).
            
        height_integrated_rayleighs_6300 (float): 
            Height-integrated Rayleighs value for the 630.0nm emission(red).
        
        height_integrated_rayleighs_8446 (float): 
            Height-integrated Rayleighs value for the 844.6nm emission (near infrared).

        altitudes (ndarray): 
            1-dimensional numpy array for the altitudes in kilometers.

        emission_4278 (ndarray): 
            1-dimensional numpy array for the 427.8nm volume emission rate (1/cm^3/s).

        emission_5577 (ndarray): 
            1-dimensional numpy array for the 557.7nm volume emission rate (1/cm^3/s).

        emission_6300 (ndarray): 
            1-dimensional numpy array for the 630.0nm volume emission rate (1/cm^3/s).

        emission_8446 (ndarray): 
            1-dimensional numpy array for the 844.6nm volume emission rate (1/cm^3/s).

        plasma_electron_density (ndarray): 
            1-dimensional numpy array for the plasma electron density (cm^-3).

        plasma_o2plus_density (ndarray): 
            1-dimensional numpy array for the plasma O2+ density (cm^-3).

        plasma_noplus_density (ndarray): 
            1-dimensional numpy array for the plasma NO+ density (cm^-3).

        plasma_oplus_density (ndarray): 
            1-dimensional numpy array for the plasma O+ density (cm^-3).

        plasma_ionisation_rate (ndarray): 
            1-dimensional numpy array for the plasma ionisation rate (1/cm^3/s).

        plasma_electron_temperature (ndarray): 
            1-dimensional numpy array for the plasma electron temperature (Kelvin).

        plasma_ion_temperature (ndarray): 
            1-dimensional numpy array for the plasma ion temperature (Kelvin).

        plasma_pederson_conductivity (ndarray): 
            1-dimensional numpy array for the Peterson plasma conductivity (S/m).

        plasma_hall_conductivity (ndarray): 
            1-dimensional numpy array for the hall plasma conductivity (S/m).

        neutral_o2_density (ndarray): 
            1-dimensional numpy array for the neutral O2 density (cm^-3).

        neutral_o_density (ndarray): 
            1-dimensional numpy array for the neutral O density (cm^-3).

        neutral_n2_density (ndarray): 
            1-dimensional numpy array for the neutral N2 density (cm^-3).

        neutral_n_density (ndarray): 
            1-dimensional numpy array for the neutral N density (cm^-3).

        neutral_temperature (ndarray): 
            1-dimensional numpy array for the neutral temperature (Kelvin).
    """
    request_info: ATMInverseResultRequestInfo
    energy_flux: Any
    characteristic_energy: Any
    oxygen_correction_factor: Any
    height_integrated_rayleighs_4278: Any
    height_integrated_rayleighs_5577: Any
    height_integrated_rayleighs_6300: Any
    height_integrated_rayleighs_8446: Any
    altitudes: Any
    emission_4278: Any
    emission_5577: Any
    emission_6300: Any
    emission_8446: Any
    plasma_electron_density: Any
    plasma_o2plus_density: Any
    plasma_noplus_density: Any
    plasma_oplus_density: Any
    plasma_ionisation_rate: Any
    plasma_electron_temperature: Any
    plasma_ion_temperature: Any
    plasma_pederson_conductivity: Any
    plasma_hall_conductivity: Any
    neutral_o2_density: Any
    neutral_o_density: Any
    neutral_n2_density: Any
    neutral_n_density: Any
    neutral_temperature: Any

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
                    var_str = "%s ...])" % (var_value.__repr__()[0:60])

            # print string for this var
            print("  %-37s: %s" % (var_name, var_str))
