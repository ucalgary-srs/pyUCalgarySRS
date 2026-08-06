Version 1.26.3 (2026-08-06)
-------------------
- updated SMILE ASI reading function to handle uncommon edge-case when loading data


Version 1.26.2 (2026-02-26)
-------------------
- updated TREx Spectrograph reading function to handle specific corrupt metadata nicer


Version 1.26.1 (2026-01-31)
-------------------
- updated `pretty_print()` function for ATM forward result


Version 1.26.0 (2026-01-31)
-------------------
- ATM model changes
  - removed support for use of ATM model version 1.0. To use this version of the model, please use a previous version of this library.
  - forward function
    - renamed `height_integrated_rayleighs_lbh` output flag to `height_integrated_rayleighs_smile_uvi_lbh`
    - renamed `emission_lbh` output flag to `emission_smile_uvi_lbh`
    - renamed `plasma_pederson_conductivity` output flag to `plasma_pedersen_conductivity`
  - inversion function
    - removed the `atmospheric_attenuation_correction` parameter (was deprecated in v1.23.0)


Version 1.25.0 (2025-12-10)
-------------------
- deprecated support for Python 3.9
- minor test suite updates


Version 1.24.0 (2025-09-11)
-------------------
- ATM inversion changes
  - renamed `characteristic_energy` output flag and data to `mean_energy`
  - added `special_logic_keyword` parameter to help handle specific non-standard use cases on the backend ATM API


Version 1.23.3 (2025-07-02)
-------------------
- bugfixes for ATM forward custom spectrum and custom neutral profile parameters


Version 1.23.2 (2025-07-02)
-------------------
- updates to ATM inversion docstrings, `pretty_print()`, `__str__()` and `__repr__()` functions


Version 1.23.1 (2025-06-21)
-------------------
- bugfix for ATM output flags


Version 1.23.0 (2025-06-21)
-------------------
- added support for TREx ATM model version 2


Version 1.22.1 (2025-06-10)
-------------------
- updated numpy dependency version range


Version 1.22.0 (2025-06-09)
-------------------
- added support for SMILE ASI data


Version 1.21.2 (2025-05-14)
-------------------
- minor update to `show_data_usage()` function


Version 1.21.1 (2025-05-14)
-------------------
- minor update to path initialization


Version 1.21.0 (2025-05-14)
-------------------
- data directory will now only be created when data is downloaded, instead of at initialization
- bugfix for TREx RGB readfile when reading a mix of good and problematic H5 files


Version 1.20.1 (2025-04-15)
-------------------
- documentation updates for ATM inverse function


Version 1.20.0 (2025-02-11)
-------------------
- updated dependency version ranges


Version 1.19.0 (2025-02-11)
-------------------
- added support for Numpy 2.0


Version 1.18.0 (2025-02-03)
-------------------
- bugfix for TREx Spectrograph processed data reading


Version 1.17.0 (2025-02-01)
-------------------
- minor update for HSR data reading


Version 1.16.0 (2025-01-31)
-------------------
- improved read performance when using the `start_time` and `end_time` parameters
- added `pretty_print()` method to `GridSourceInfoData` class
- updated test suite
- bugfixes for data reading edge cases; SWAN HSR, NORSTAR riometer, TREx Spectrograph, skymap and calibration files


Version 1.15.0 (2025-01-26)
-------------------
- docstring updates


Version 1.14.0 (2025-01-26)
-------------------
- docstring updates


Version 1.13.0 (2025-01-23)
-------------------
- docstring updates


Version 1.12.0 (2025-01-21)
-------------------
- updated defaulting for `progress_bar_backend`
- updated warning messages for better handling inside of VSCode Jupyter extension
- updated class string methods for `FileListingResponse` and `FileDownloadResult`


Version 1.11.0 (2025-01-20)
-------------------
- added `progress_bar_backend` to the `PyUCalgarySRS()` object


Version 1.10.0 (2025-01-18)
-------------------
- minor adjustments to `api_base_url` parameter in `PyUCalgarySRS` object
- added level filtering to the `list_datasets()` function


Version 1.9.0 (2025-01-10)
-------------------
- added warning to `download()` function if no data was found to download


Version 1.8.0
-------------------
- added `include_total_bytes` parameter to `get_urls()` function


Version 1.7.1 (2024-12-29)
-------------------
- performance improvement for reading raw ASI data
- removed `joblib` dependency


Version 1.6.3 (2024-12-06)
-------------------
- further updates to handle TREx Spectrograph skymap reading
- formatting updates for `pretty_print()` functions


Version 1.6.2 (2024-12-06)
-------------------
- updates to handle TREx Spectrograph skymap reading


Version 1.6.1 (2024-12-05)
-------------------
- bugfix for riometer K2 reading


Version 1.6.0 (2024-12-05)
-------------------
- added `start_time` and `end_time` parameters to all `read()` functions
- added `file_time_resolution` attribute to Dataset objects


Version 1.5.1 (2024-12-02)
-------------------
- bugfix for `get_dataset()` function


Version 1.5.0 (2024-12-02)
-------------------
- added support for downloading, reading, and analysis of the TREx Spectrograph data
- added `get_dataset()` function for retrieving a specific single dataset


Version 1.4.0 (2024-11-29)
-------------------
- added `supported_library` attribute to Dataset objects


Version 1.3.0 - 1.3.4 (2024-11-28)
-------------------
- added several `pretty_print()` functions for classes
- added riometer and HSR readfile routines


Version 1.2.0 (2024-07-09)
-------------------
- default ATM transport timescale changed from 300 to 600


Version 1.0.1 to 1.1.1 (2024-06-26)
--------------------
Various bugfixes and minor tweaks.


Version 1.0.0 (2024-06-17)
--------------------
Initial stable release.
