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
Functions for running TREx Auroral Transport Model (ATM) calculations.
"""

import datetime
from .classes_forward import *
from .classes_inverse import *
from ._forward import forward as func_forward
from ._inverse import inverse as func_inverse

ATM_DEFAULT_MAXWELLIAN_ENERGY_FLUX = 10.0
ATM_DEFAULT_GAUSSIAN_ENERGY_FLUX = 0.0
ATM_DEFAULT_MAXWELLIAN_CHARACTERISTIC_ENERGY = 5000.0
ATM_DEFAULT_GAUSSIAN_PEAK_ENERGY = 1000.0
ATM_DEFAULT_GAUSSIAN_SPECTRAL_WIDTH = 100.0
ATM_DEFAULT_NRLMSIS_MODEL_VERSION = "2.0"
ATM_DEFAULT_OXYGEN_CORRECTION_FACTOR = 1.0
ATM_DEFAULT_TIMESCALE_AURORAL = 600
ATM_DEFAULT_TIMESCALE_TRANSPORT = 300
ATM_DEFAULT_MODEL_VERSION = "1.0"
ATM_DEFAULT_PRECIPITATION_SPECTRAL_FLUX_TYPE = "gaussian"


class ATMManager:
    """
    The ATMManager object is initialized within every PyUCalgarySRS.models object. It 
    acts as a way to access the submodules and carry over configuration information in 
    the super class.
    """

    def __init__(self, srs_obj):
        self.__srs_obj = srs_obj

    def forward(self,
                timestamp: datetime.datetime,
                geodetic_latitude: float,
                geodetic_longitude: float,
                output: ATMForwardOutputFlags,
                maxwellian_energy_flux: float = ATM_DEFAULT_MAXWELLIAN_ENERGY_FLUX,
                gaussian_energy_flux: float = ATM_DEFAULT_GAUSSIAN_ENERGY_FLUX,
                maxwellian_characteristic_energy: float = ATM_DEFAULT_MAXWELLIAN_CHARACTERISTIC_ENERGY,
                gaussian_peak_energy: float = ATM_DEFAULT_GAUSSIAN_PEAK_ENERGY,
                gaussian_spectral_width: float = ATM_DEFAULT_GAUSSIAN_SPECTRAL_WIDTH,
                nrlmsis_model_version: Literal["00", "2.0"] = ATM_DEFAULT_NRLMSIS_MODEL_VERSION,
                oxygen_correction_factor: float = ATM_DEFAULT_OXYGEN_CORRECTION_FACTOR,
                timescale_auroral: int = ATM_DEFAULT_TIMESCALE_AURORAL,
                timescale_transport: int = ATM_DEFAULT_TIMESCALE_TRANSPORT,
                atm_model_version: Literal["1.0"] = ATM_DEFAULT_MODEL_VERSION,
                custom_spectrum: Optional[ndarray] = None,
                no_cache: bool = False,
                timeout: Optional[int] = None) -> ATMForwardResult:
        """
        Perform a forward calculation using the TREx Auroral Transport Model and the supplied input 
        parameters. Note that this function utilizes the UCalgary Space Remote Sensing API to perform 
        the calculation.

        Args:
            timestamp (datetime.datetime): 
                Timestamp for the calculation. This value is expected to be in UTC, and is valid for 
                any value up to the end of the previous day. Any timezone data will be ignored. This 
                parameter is required.

            geodetic_latitude (float): 
                Latitude in geodetic coordinates: -90.0 to 90.0. This parameter is required.

            geodetic_longitude (float): 
                Longitude in geodetic coordinates: -180.0 to 180.0. This parameter is required.

            output (ATMForwardOutputFlags): 
                Flags to indicate which values are included in the output. See 
                [`ATMForwardOutputFlags`](classes_forward.html#pyucalgarysrs.models.atm.classes_forward.ATMForwardOutputFlags) 
                for more details. This parameter is required.

            maxwellian_energy_flux (float): 
                Maxwellian energy flux in erg/cm2/s. Default is 10. This parameter is optional.

            gaussian_energy_flux (float): 
                Gaussian energy flux in erg/cm2/s. Default is 0.0. Note that `gaussian_peak_energy` and
                `gaussian_spectral_width` must be specified if the `gaussian_energy_flux` is not 0. 
                This parameter is optional.

            maxwellian_characteristic_energy (float): 
                Maxwellian characteristic energy in eV. Default is 5000. Note that `maxwellian_characteristic_energy` 
                must be specified if the `maxwellian_energy_flux` is not 0. This parameter 
                is optional.

            gaussian_peak_energy (float): 
                Gaussian peak energy in eV. Default is 1000. Note this parameter must be specified if 
                the `gaussian_energy_flux` is not 0. This parameter is optional.

            gaussian_spectral_width (float): 
                Gaussian spectral width in eV. Default is 100. Note this parameter must be specified if 
                the `gaussian_energy_flux` is not 0. This parameter is optional.

            nrlmsis_model_version (str): 
                NRLMSIS version number. Possible values are `00` or `2.0`. Default is `2.0`. This parameter is
                optional. More details about this empirical model can be found [here](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~00/),
                and [here](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~2.0/).

            oxygen_correction_factor (float): 
                Oxygen correction factor used to multiply by in the empirical model. Default is 1. This parameter
                is optional.

            timescale_auroral (int): 
                Auroral timescale in seconds. Default is 600 (10 minutes). This parameter is optional.

            timescale_transport (int): 
                Transport timescale in seconds. Default is 300 (5 minutes). This parameter is optional.

            atm_model_version (str): 
                ATM model version number. Possible values are only '1.0' at this time, but will have
                additional possible values in the future. This parameter is optional.

            custom_spectrum (ndarray): 
                A 2-dimensional numpy array (dtype is any float type) containing values representing the
                energy in eV, and flux in 1/cm2/sr/eV. The shape is expected to be [N, 2], with energy in
                [:, 0] and flux in [:, 1]. Note that this array cannot contain negative values (SRSAPIError 
                will be raised if so). This parameter is optional.

            no_cache (bool): 
                The UCalgary Space Remote Sensing API utilizes a caching layer for performing ATM
                calculations. If this variation of input parameters has been run before (and the
                cache is still valid), then it will not re-run the calculation. Instead it will 
                return the cached results immediately. To disable the caching layer, set this 
                parameter to `True`. Default is `False`. This parameter is optional.

            timeout (int): 
                Represents how many seconds to wait for the API to send data before giving up. The 
                default is 30 seconds, or the `api_timeout` value in the super class' `pyucalgarysrs.PyUCalgarySRS`
                object. This parameter is optional.
        
        Returns:
            An [`ATMForwardResult`](classes_forward.html#pyucalgarysrs.models.atm.classes_forward.ATMForwardResult)
            object containing the requested output data, among other values.

        Raises:
            pyucalgarysrs.exceptions.SRSAPIError: An API error was encountered
        """
        return func_forward(
            self.__srs_obj,
            timestamp,
            geodetic_latitude,
            geodetic_longitude,
            output,
            maxwellian_energy_flux,
            gaussian_energy_flux,
            maxwellian_characteristic_energy,
            gaussian_peak_energy,
            gaussian_spectral_width,
            nrlmsis_model_version,
            oxygen_correction_factor,
            timescale_auroral,
            timescale_transport,
            atm_model_version,
            custom_spectrum,
            no_cache,
            timeout,
        )

    def inverse(self,
                timestamp: datetime.datetime,
                geodetic_latitude: float,
                geodetic_longitude: float,
                intensity_4278: float,
                intensity_5577: float,
                intensity_6300: float,
                intensity_8446: float,
                output: ATMInverseOutputFlags,
                precipitation_flux_spectral_type: Literal["gaussian", "maxwellian"] = ATM_DEFAULT_PRECIPITATION_SPECTRAL_FLUX_TYPE,
                nrlmsis_model_version: Literal["00", "2.0"] = ATM_DEFAULT_NRLMSIS_MODEL_VERSION,
                atm_model_version: Literal["1.0"] = ATM_DEFAULT_MODEL_VERSION,
                no_cache: bool = False,
                timeout: Optional[int] = None) -> ATMInverseResult:
        """
        Perform an inverse calculation using the TREx Auroral Transport Model and the supplied input 
        parameters. Note that this function utilizes the UCalgary Space Remote Sensing API to perform 
        the calculation.

        Args:
            timestamp (datetime.datetime): 
                Timestamp for the calculation. This value is expected to be in UTC, and is valid for a pre-defined 
                timeframe. An error will be raised if outside of the valid timeframe. Any timezone data will be 
                ignored. This parameter is required.

            geodetic_latitude (float): 
                Latitude in geodetic coordinates. Currently limited to the Transition Region Explorer (TREx)
                region of >=50.0 and <61.5 degrees. An error will be raised if outside of this range. This 
                parameter is required.

            geodetic_longitude (float): 
                Longitude in geodetic coordinates. Currently limited to the Transition Region Explorer (TREx)
                region of >=-110 and <-70 degrees. An error will be raised if outside of this range. This 
                parameter is required.

            intensity_4278 (float): 
                Intensity of the 427.8nm (blue) wavelength in Rayleighs. This parameter is required.                

            intensity_5577 (float): 
                Intensity of the 557.7nm (green) wavelength in Rayleighs. This parameter is required.                

            intensity_6300 (float): 
                Intensity of the 630.0nm (red) wavelength in Rayleighs. This parameter is required.                

            intensity_8446 (float): 
                Intensity of the 844.6nm (near infrared) wavelength in Rayleighs. This parameter is required.                

            output (ATMInverseOutputFlags): 
                Flags to indicate which values are included in the output. See 
                [`ATMInverseOutputFlags`](classes_inverse.html#pyucalgarysrs.models.atm.classes_inverse.ATMInverseOutputFlags) 
                for more details. This parameter is required.

            precipitation_flux_spectral_type (str): 
                The precipitation flux spectral type to use. Possible values are `gaussian` or `maxwellian`. The
                default is `gaussian`. This parameter is optional.

            nrlmsis_model_version (str): 
                NRLMSIS version number. Possible values are `00` or `2.0`. Default is `2.0`. This parameter is
                optional. More details about this empirical model can be found [here](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~00/),
                and [here](https://ccmc.gsfc.nasa.gov/models/NRLMSIS~2.0/).

            atm_model_version (str): 
                ATM model version number. Possible values are only '1.0' at this time, but will have
                additional possible values in the future. This parameter is optional.

            no_cache (bool): 
                The UCalgary Space Remote Sensing API utilizes a caching layer for performing ATM
                calculations. If this variation of input parameters has been run before (and the
                cache is still valid), then it will not re-run the calculation. Instead it will 
                return the cached results immediately. To disable the caching layer, set this 
                parameter to `True`. Default is `False`. This parameter is optional.

            timeout (int): 
                Represents how many seconds to wait for the API to send data before giving up. The 
                default is 30 seconds, or the `api_timeout` value in the super class' `pyucalgarysrs.PyUCalgarySRS`
                object. This parameter is optional.

        Returns:
            An [`ATMInverseResult`](classes_inverse.html#pyucalgarysrs.models.atm.classes_inverse.ATMInverseResult)
            object containing the requested output data, among other values.

        Raises:
            pyucalgarysrs.exceptions.SRSAPIError: An API error was encountered
        """
        return func_inverse(
            self.__srs_obj,
            timestamp,
            geodetic_latitude,
            geodetic_longitude,
            intensity_4278,
            intensity_5577,
            intensity_6300,
            intensity_8446,
            output,
            precipitation_flux_spectral_type,
            nrlmsis_model_version,
            atm_model_version,
            no_cache,
            timeout,
        )
