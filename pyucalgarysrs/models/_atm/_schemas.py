import datetime
from dataclasses import dataclass
from typing import Optional, Literal
from numpy import ndarray


@dataclass
class ATMForwardOutputFlags:
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

    def set_all_true(self):
        excluded_items = ["set_all_true", "set_all_false", "enable_only_height_integrated_rayleighs"]
        for var_name in dir(self):
            if (not var_name.startswith("__") and var_name not in excluded_items):
                setattr(self, var_name, True)

    def set_all_false(self):
        excluded_items = ["set_all_true", "set_all_false", "enable_only_height_integrated_rayleighs"]
        for var_name in dir(self):
            if (not var_name.startswith("__") and var_name not in excluded_items):
                setattr(self, var_name, False)

    def enable_only_height_integrated_rayleighs(self):
        for var_name in filter(lambda x: x.startswith("height_integrated_rayleighs"), dir(self)):
            setattr(self, var_name, True)


@dataclass
class ATMForwardRequest:
    atm_model_version: Literal["1.0"]
    timestamp: datetime.datetime
    geodetic_latitude: float
    geodetic_longitude: float
    maxwellian_energy_flux: float
    gaussian_energy_flux: float
    maxwellian_characteristic_energy: float
    gaussian_peak_energy: float
    gaussian_spectral_width: float
    nrlmsis_model_version: Literal["00", "2.0"]
    oxygen_correction_factor: float
    timescale_auroral: int
    timescale_transport: int
    custom_spectrum: Optional[ndarray]
    output: ATMForwardOutputFlags
    no_cache: bool


@dataclass
class ATMForwardResultRequestInfo:
    request: ATMForwardRequest
    calculation_duration_ms: float


@dataclass
class ATMForwardResult:
    request_info: ATMForwardResultRequestInfo
    height_integrated_rayleighs_4278: Optional[float] = None
    height_integrated_rayleighs_5577: Optional[float] = None
    height_integrated_rayleighs_6300: Optional[float] = None
    height_integrated_rayleighs_8446: Optional[float] = None
    height_integrated_rayleighs_lbh: Optional[float] = None
    height_integrated_rayleighs_1304: Optional[float] = None
    height_integrated_rayleighs_1356: Optional[float] = None
    altitudes: Optional[ndarray] = None
    emission_4278: Optional[ndarray] = None
    emission_5577: Optional[ndarray] = None
    emission_6300: Optional[ndarray] = None
    emission_8446: Optional[ndarray] = None
    emission_lbh: Optional[ndarray] = None
    emission_1304: Optional[ndarray] = None
    emission_1356: Optional[ndarray] = None
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
        print("ATMForwardResult:")
        for var_name in dir(self):
            # exclude methods
            if (var_name.startswith("__") or var_name == "pretty_print"):
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
            print("  %-35s: %s" % (var_name, var_str))
