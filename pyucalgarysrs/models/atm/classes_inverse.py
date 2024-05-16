"""
Classes for representing ATM inverse calculation requests and responses. All
classes in this module are included at the top level of this library.
"""

import datetime
from dataclasses import dataclass
from typing import Optional, Literal
from numpy import ndarray


@dataclass
class ATMInverseOutputFlags:
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
        excluded_items = ["set_all_true", "set_all_false"]
        for var_name in dir(self):
            if (not var_name.startswith("__") and var_name not in excluded_items):
                setattr(self, var_name, True)

    def set_all_false(self):
        excluded_items = ["set_all_true", "set_all_false"]
        for var_name in dir(self):
            if (not var_name.startswith("__") and var_name not in excluded_items):
                setattr(self, var_name, False)


@dataclass
class ATMInverseRequest:
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
    output: ATMInverseOutputFlags
    no_cache: bool


@dataclass
class ATMInverseForwardParams:
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
    request: ATMInverseRequest
    calculation_duration_ms: float
    forward_params: Optional[ATMInverseForwardParams] = None


@dataclass
class ATMInverseResult:
    request_info: ATMInverseResultRequestInfo
    energy_flux: Optional[float] = None
    characteristic_energy: Optional[float] = None
    oxygen_correction_factor: Optional[float] = None
    height_integrated_rayleighs_4278: Optional[float] = None
    height_integrated_rayleighs_5577: Optional[float] = None
    height_integrated_rayleighs_6300: Optional[float] = None
    height_integrated_rayleighs_8446: Optional[float] = None
    altitudes: Optional[ndarray] = None
    emission_4278: Optional[ndarray] = None
    emission_5577: Optional[ndarray] = None
    emission_6300: Optional[ndarray] = None
    emission_8446: Optional[ndarray] = None
    plasma_electron_density: Optional[ndarray] = None
    plasma_o2plus_density: Optional[ndarray] = None
    plasma_noplus_density: Optional[ndarray] = None
    plasma_oplus_density: Optional[ndarray] = None
    plasma_ionisation_rate: Optional[ndarray] = None
    plasma_electron_temperature: Optional[ndarray] = None
    plasma_ion_temperature: Optional[ndarray] = None
    plasma_pederson_conductivity: Optional[ndarray] = None
    plasma_hall_conductivity: Optional[ndarray] = None
    neutral_o2_density: Optional[ndarray] = None
    neutral_o_density: Optional[ndarray] = None
    neutral_n2_density: Optional[ndarray] = None
    neutral_n_density: Optional[ndarray] = None
    neutral_temperature: Optional[ndarray] = None

    def pretty_print(self):
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
            print("  %-35s: %s" % (var_name, var_str))
