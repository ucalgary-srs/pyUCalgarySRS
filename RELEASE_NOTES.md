Version 1.26.1
-------------------
- updated `pretty_print()` function for ATM forward result


Version 1.26.0
-------------------
- ATM model changes
  - removed support for use of ATM model version 1.0. To use this version of the model, please use a previous version of this library.
  - forward function
    - renamed `height_integrated_rayleighs_lbh` output flag to `height_integrated_rayleighs_smile_uvi_lbh`
    - renamed `emission_lbh` output flag to `emission_smile_uvi_lbh`
    - renamed `plasma_pederson_conductivity` output flag to `plasma_pedersen_conductivity`
  - inversion function
    - removed the `atmospheric_attenuation_correction` parameter (was deprecated in v1.23.0)


Version 1.25.0
-------------------
- deprecated support for Python 3.9
- minor test suite updates


Version 1.24.0
-------------------
- ATM inversion changes
  - renamed `characteristic_energy` output flag and data to `mean_energy`
  - added `special_logic_keyword` parameter to help handle specific non-standard use cases on the backend ATM API


Version 1.23.3
-------------------
- bugfixes for ATM forward custom spectrum and custom neutral profile parameters


Version 1.23.2
-------------------
- updates to ATM inversion docstrings, `pretty_print()`, `__str__()` and `__repr__()` functions


Version 1.23.1
-------------------
- bugfix for ATM output flags


Version 1.23.0
-------------------
- added support for TREx ATM model version 2


Version 1.22.1
-------------------
- updated numpy dependency version range


Version 1.22.0
-------------------
- added support for SMILE ASI data


Version 1.21.2
-------------------
- minor update to `show_data_usage()` function


Version 1.21.1
-------------------
- minor update to path initialization


Version 1.21.0
-------------------
- data directory will now only be created when data is downloaded, instead of at initialization
- bugfix for TREx RGB readfile when reading a mix of good and problematic H5 files


Version 1.20.1
-------------------
- documentation updates for ATM inverse function


Version 1.20.0
-------------------
- updated dependency version ranges


Version 1.19.0
-------------------
- added support for Numpy 2.0


Version 1.18.0
-------------------
- bugfix for TREx Spectrograph processed data reading


Version 1.17.0
-------------------
- minor update for HSR data reading


Version 1.16.0
-------------------
- improved read performance when using the `start_time` and `end_time` parameters
- added `pretty_print()` method to `GridSourceInfoData` class
- updated test suite
- bugfixes for data reading edge cases; SWAN HSR, NORSTAR riometer, TREx Spectrograph, skymap and calibration files


Version 1.15.0
-------------------
- docstring updates


Version 1.14.0
-------------------
- docstring updates


Version 1.13.0
-------------------
- docstring updates


Version 1.12.0
-------------------
- updated defaulting for `progress_bar_backend`
- updated warning messages for better handling inside of VSCode Jupyter extension
- updated class string methods for `FileListingResponse` and `FileDownloadResult`


Version 1.11.0
-------------------
- added `progress_bar_backend` to the `PyUCalgarySRS()` object


Version 1.10.0
-------------------
- minor adjustments to `api_base_url` parameter in `PyUCalgarySRS` object
- added level filtering to the `list_datasets()` function


Version 1.9.0
-------------------
- added warning to `download()` function if no data was found to download


Version 1.8.0
-------------------
- added `include_total_bytes` parameter to `get_urls()` function


Version 1.7.1
-------------------
- performance improvement for reading raw ASI data
- removed `joblib` dependency


Version 1.6.3
-------------------
- further updates to handle TREx Spectrograph skymap reading
- formatting updates for `pretty_print()` functions


Version 1.6.2
-------------------
- updates to handle TREx Spectrograph skymap reading


Version 1.6.1
-------------------
- bugfix for riometer K2 reading


Version 1.6.0
-------------------
- added `start_time` and `end_time` parameters to all `read()` functions
- added `file_time_resolution` attribute to Dataset objects


Version 1.5.1
-------------------
- bugfix for `get_dataset()` function


Version 1.5.0
-------------------
- added support for downloading, reading, and analysis of the TREx Spectrograph data
- added `get_dataset()` function for retrieving a specific single dataset


Version 1.4.0
-------------------
- added `supported_library` attribute to Dataset objects


Version 1.3.0 - 1.3.4
-------------------
- added several `pretty_print()` functions for classes
- added riometer and HSR readfile routines


Version 1.2.0
-------------------
- default ATM transport timescale changed from 300 to 600


Version 1.0.1 to 1.1.1
--------------------
Various bugfixes and minor tweaks.


Version 1.0.0
--------------------
Initial stable release.
