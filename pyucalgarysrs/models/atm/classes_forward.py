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
Classes for representing ATM forward calculation requests and responses. All
classes in this module are included at the top level of this library.
"""

import datetime
from dataclasses import dataclass
from typing import Optional, Literal, Any
from numpy import ndarray


@dataclass
class ATMForwardOutputFlags:
    """
    Class to represent all output values included in an ATM forward calculation.
    ATM calculations are performed in a way where you can toggle ON/OFF whichever
    pieces of information you do or don't want. This improves efficiency of the 
    calculation routine resulting in faster queries.

    By default, all output flags are set to False. There exist several helper methods
    to toggle all to True, toggle all to False, or toggle only height-integrated Rayleighs 
    flags to True. See the below Methods section for details.

    Details about each output value can be found in the documentation for the 
    [`ATMForwardResult`](classes_forward.html#pyucalgarysrs.models.atm.classes_forward.ATMForwardResult)
    object.
    """
    height_integrated_rayleighs_4278: bool = False
    height_integrated_rayleighs_5577: bool = False
    height_integrated_rayleighs_6300: bool = False
    height_integrated_rayleighs_8446: bool = False
    height_integrated_rayleighs_lbh: bool = False
    height_integrated_rayleighs_1304: bool = False
    height_integrated_rayleighs_1356: bool = False
    altitudes: bool = False
    emission_4278: bool = False
    emission_5577: bool = False
    emission_6300: bool = False
    emission_8446: bool = False
    emission_lbh: bool = False
    emission_1304: bool = False
    emission_1356: bool = False
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
    production_rate_o2plus: bool = False
    production_rate_oplus: bool = False
    production_rate_oneg: bool = False
    production_rate_o: bool = False
    production_rate_nplus: bool = False
    production_rate_n2plus: bool = False
    production_rate_n: bool = False
    production_rate_n2d: bool = False

    def set_all_true(self):
        """
        Sets all flags to `True`.
        """
        excluded_items = ["set_all_true", "set_all_false", "enable_only_height_integrated_rayleighs"]
        for var_name in dir(self):
            if (not var_name.startswith("__") and var_name not in excluded_items):
                setattr(self, var_name, True)

    def set_all_false(self):
        """
        Sets all flags to `False`.
        """
        excluded_items = ["set_all_true", "set_all_false", "enable_only_height_integrated_rayleighs"]
        for var_name in dir(self):
            if (not var_name.startswith("__") and var_name not in excluded_items):
                setattr(self, var_name, False)

    def enable_only_height_integrated_rayleighs(self):
        """
        Sets only height-integrated Rayleighs values to `True`.
        """
        for var_name in filter(lambda x: x.startswith("height_integrated_rayleighs"), dir(self)):
            setattr(self, var_name, True)


@dataclass
class ATMForwardRequest:
    """
    Class that represents the UCalgary Space Remote Sensing API request when 
    performing an ATM forward calculation. This object is included in any
    [`ATMForwardResultRequestInfo`](classes_forward.html#pyucalgarysrs.models.atm.classes_forward.ATMForwardResultRequestInfo) 
    and [`ATMForwardResult`](classes_forward.html#pyucalgarysrs.models.atm.classes_forward.ATMForwardResult) 
    objects.
    """
    atm_model_version: Literal["1.0", "2.0"]
    timestamp: datetime.datetime
    geodetic_latitude: float
    geodetic_longitude: float
    maxwellian_energy_flux: float
    maxwellian_characteristic_energy: float
    gaussian_energy_flux: float
    gaussian_peak_energy: float
    gaussian_spectral_width: float
    kappa_energy_flux: float
    kappa_mean_energy: float
    kappa_k_index: float
    exponential_energy_flux: float
    exponential_characteristic_energy: float
    exponential_starting_energy: float
    proton_energy_flux: float
    proton_characteristic_energy: float
    d_region: bool
    nrlmsis_model_version: Literal["00", "2.0"]
    oxygen_correction_factor: float
    custom_spectrum: Optional[ndarray]
    custom_neutral_profile: Optional[ndarray]
    timescale_auroral: int
    timescale_transport: int
    output: ATMForwardOutputFlags
    no_cache: bool


@dataclass
class ATMForwardResultRequestInfo:
    """
    Class containing details about interacting with the UCalgary Space Remote Sensing
    API when performing an ATM forward calculation. It contains the API request itself,
    and other information returned by the API.

    An instance of this is included in any
    [`ATMForwardResult`](classes_forward.html#pyucalgarysrs.models.atm.classes_forward.ATMForwardResult) 
    object.

    Attributes:
        request (ATMForwardRequest): 
            Instance of class that represents the lower-level API request when performing 
            an ATM forward calculation.

        calculation_duration_ms (float): 
            Duration the the API spent performing the ATM forward calculation. Represented
            in milliseconds.
    """
    request: ATMForwardRequest
    calculation_duration_ms: float


@dataclass
class ATMForwardResult:
    """
    Class containing all data from an ATM forward calculation. This class also includes
    details about the input parameters to the forward calculation routine (`request_info`),
    along with all output data values.
    
    Based on the request's output flags, if flags were set to False then the values will 
    be None types. If flags were set to True then the values will be their respective 
    types (e.g., float, numpy ndarray).

    Attributes:
        request_info (ATMForwardResultRequestInfo): 
            Information about the API request made to perform the ATM forward calculation.
        
        height_integrated_rayleighs_4278 (float): 
            Height-integrated Rayleighs value for the 427.8nm emission (blue).

        height_integrated_rayleighs_5577 (float): 
            Height-integrated Rayleighs value for the 557.7nm emission (green).
            
        height_integrated_rayleighs_6300 (float): 
            Height-integrated Rayleighs value for the 630.0nm emission(red).
        
        height_integrated_rayleighs_8446 (float): 
            Height-integrated Rayleighs value for the 844.6nm emission (near infrared).

        height_integrated_rayleighs_lbh (float): 
            Height-integrated Rayleighs value for the Lyman-Birge-Hopfield emission.

        height_integrated_rayleighs_1304 (float): 
            Height-integrated Rayleighs value for the 130.4nm emission.

        height_integrated_rayleighs_1356 (float): 
            Height-integrated Rayleighs value for the 135.6nm emission.

        altitudes (ndarray): 
            A 1-dimensional numpy array for the altitudes in kilometers.

        emission_4278 (ndarray): 
            A 1-dimensional numpy array for the 427.8nm volume emission rate (1/cm^3/s).

        emission_5577 (ndarray): 
            A 1-dimensional numpy array for the 557.7nm volume emission rate (1/cm^3/s).

        emission_6300 (ndarray): 
            A 1-dimensional numpy array for the 630.0nm volume emission rate (1/cm^3/s).

        emission_8446 (ndarray): 
            A 1-dimensional numpy array for the 844.6nm volume emission rate (1/cm^3/s).

        emission_lbh (ndarray): 
            A 1-dimensional numpy array for the Lyman-Birge-Hopfield volume emission rate (1/cm^3/s).

        emission_1304 (ndarray): 
            A 1-dimensional numpy array for the 130.4nm volume emission rate (1/cm^3/s).

        emission_1356 (ndarray): 
            A 1-dimensional numpy array for the 135.6nm volume emission rate (1/cm^3/s).

        plasma_electron_density (ndarray): 
            A 1-dimensional numpy array for the plasma electron density (cm^-3).

        plasma_o2plus_density (ndarray): 
            A 1-dimensional numpy array for the plasma O2+ density (cm^-3).

        plasma_noplus_density (ndarray): 
            A 1-dimensional numpy array for the plasma NO+ density (cm^-3).

        plasma_oplus_density (ndarray): 
            A 1-dimensional numpy array for the plasma O+ density (cm^-3).

        plasma_ionisation_rate (ndarray): 
            A 1-dimensional numpy array for the plasma ionisation rate (1/cm^3/s).

        plasma_electron_temperature (ndarray): 
            A 1-dimensional numpy array for the plasma electron temperature (Kelvin).

        plasma_ion_temperature (ndarray): 
            A 1-dimensional numpy array for the plasma ion temperature (Kelvin).

        plasma_pederson_conductivity (ndarray): 
            A 1-dimensional numpy array for the Peterson plasma conductivity (S/m).

        plasma_hall_conductivity (ndarray): 
            A 1-dimensional numpy array for the hall plasma conductivity (S/m).

        neutral_o2_density (ndarray): 
            A 1-dimensional numpy array for the neutral O2 density (cm^-3).

        neutral_o_density (ndarray): 
            A 1-dimensional numpy array for the neutral O density (cm^-3).

        neutral_n2_density (ndarray): 
            A 1-dimensional numpy array for the neutral N2 density (cm^-3).

        neutral_n_density (ndarray): 
            A 1-dimensional numpy array for the neutral N density (cm^-3).

        neutral_temperature (ndarray): 
            A 1-dimensional numpy array for the neutral temperature (Kelvin).

        production_rate_o2plus (ndarray): 
            A 1-dimensional numpy array for the O2+ production rate (1/cm^3/s)

        production_rate_oplus (ndarray): 
            A 1-dimensional numpy array for the O+ production rate (1/cm^3/s)

        production_rate_oneg (ndarray): 
            A 1-dimensional numpy array for the O- production rate (1/cm^3/s)

        production_rate_o (ndarray): 
            A 1-dimensional numpy array for the O production rate (1/cm^3/s)

        production_rate_nplus (ndarray): 
            A 1-dimensional numpy array for the N+ production rate (1/cm^3/s)

        production_rate_n2plus (ndarray): 
            A 1-dimensional numpy array for the N2+ production rate (1/cm^3/s)

        production_rate_n (ndarray): 
            A 1-dimensional numpy array for the N production rate (1/cm^3/s)

        production_rate_n2d (ndarray): 
            A 1-dimensional numpy array for the N(2d) production rate (1/cm^3/s)
    """
    request_info: ATMForwardResultRequestInfo
    height_integrated_rayleighs_4278: Any
    height_integrated_rayleighs_5577: Any
    height_integrated_rayleighs_6300: Any
    height_integrated_rayleighs_8446: Any
    height_integrated_rayleighs_lbh: Any
    height_integrated_rayleighs_1304: Any
    height_integrated_rayleighs_1356: Any
    altitudes: Any
    emission_4278: Any
    emission_5577: Any
    emission_6300: Any
    emission_8446: Any
    emission_lbh: Any
    emission_1304: Any
    emission_1356: Any
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
    production_rate_o2plus: Any
    production_rate_oplus: Any
    production_rate_oneg: Any
    production_rate_o: Any
    production_rate_nplus: Any
    production_rate_n2plus: Any
    production_rate_n: Any
    production_rate_n2d: Any

    def pretty_print(self):
        """
        A special print output for this class.
        """
        print("ATMForwardResult:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
                continue

            # exclude based on version
            if (self.request_info.request.atm_model_version == "1.0" and "production_rate_" in var_name):
                continue

            # convert var to string format we want
            var_value = getattr(self, var_name)
            var_str = "None"
            if (var_name == "request_info"):
                var_str = "ATMForwardResultRequestInfo(...)"
            if (var_value is not None):
                if (isinstance(var_value, float)):
                    var_str = "%f" % (var_value)
                elif (isinstance(var_value, ndarray)):
                    var_str = "%s ...])" % (var_value.__repr__()[0:60])

            # print string for this var
            print("  %-37s: %s" % (var_name, var_str))
