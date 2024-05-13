import datetime
from ._forward import forward as func_forward
from ._inverse import inverse as func_inverse
from ._schemas import *


class ATMManager:

    __ATM_DEFAULT_MAXWELLIAN_ENERGY_FLUX = 10.0
    __ATM_DEFAULT_GAUSSIAN_ENERGY_FLUX = 0.0
    __ATM_DEFAULT_MAXWELLIAN_CHARACTERISTIC_ENERGY = 5000.0
    __ATM_DEFAULT_GAUSSIAN_PEAK_ENERGY = 1000.0
    __ATM_DEFAULT_GAUSSIAN_SPECTRAL_WIDTH = 100.0
    __ATM_DEFAULT_NRLMSIS_MODEL_VERSION = "2.0"
    __ATM_DEFAULT_OXYGEN_CORRECTION_FACTOR = 1.0
    __ATM_DEFAULT_TIMESCALE_AURORAL = 600
    __ATM_DEFAULT_TIMESCALE_TRANSPORT = 300
    __ATM_DEFAULT_MODEL_VERSION = "1.0"

    def __init__(self, srs_obj):
        self.__srs_obj = srs_obj

    def forward(self,
                timestamp: datetime.datetime,
                geodetic_latitude: float,
                geodetic_longitude: float,
                output: ATMForwardOutputFlags,
                maxwellian_energy_flux: float = __ATM_DEFAULT_MAXWELLIAN_ENERGY_FLUX,
                gaussian_energy_flux: float = __ATM_DEFAULT_GAUSSIAN_ENERGY_FLUX,
                maxwellian_characteristic_energy: float = __ATM_DEFAULT_MAXWELLIAN_CHARACTERISTIC_ENERGY,
                gaussian_peak_energy: float = __ATM_DEFAULT_GAUSSIAN_PEAK_ENERGY,
                gaussian_spectral_width: float = __ATM_DEFAULT_GAUSSIAN_SPECTRAL_WIDTH,
                nrlmsis_model_version: Literal["00", "2.0"] = __ATM_DEFAULT_NRLMSIS_MODEL_VERSION,
                oxygen_correction_factor: float = __ATM_DEFAULT_OXYGEN_CORRECTION_FACTOR,
                timescale_auroral: int = __ATM_DEFAULT_TIMESCALE_AURORAL,
                timescale_transport: int = __ATM_DEFAULT_TIMESCALE_TRANSPORT,
                atm_model_version: Literal["1.0"] = __ATM_DEFAULT_MODEL_VERSION,
                custom_spectrum: Optional[ndarray] = None,
                no_cache: bool = False,
                timeout: Optional[int] = None) -> ATMForwardResult:
        """
        Perform a forward calculation using the TREx Auroral Transport Model and the supplied input parameters.
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

    def inverse(self) -> None:
        """
        Perform an inverse calculation using the TREx Auroral Transport Model and the supplied input parameters.
        """
        return func_inverse(self.__srs_obj)
