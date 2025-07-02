import datetime
import pyucalgarysrs
import numpy as np

srs = pyucalgarysrs.PyUCalgarySRS()

output = pyucalgarysrs.ATMForwardOutputFlags()
output.enable_only_height_integrated_rayleighs()
timestamp = datetime.datetime(2021, 11, 4, 6, 0, 0)
latitude = 53.1
longitude = -107.7

num_neut = 16
neutral_profile_arr = np.zeros((7, num_neut), order='F', dtype=np.single)
for i in range(num_neut):
    neutral_profile_arr[0, i] = 50. + i * 50.
    neutral_profile_arr[2, i] = 1e16 * np.exp(-2. * i)
    neutral_profile_arr[3, i] = 2.5e15 * np.exp(-2. * i)
    neutral_profile_arr[4, i] = (1e6) - i * 100.
    neutral_profile_arr[5, i] = 1e8 * np.exp(-1. * i)
    neutral_profile_arr[6, i] = 200. + i * 50.

    if i < 1:
        neutral_profile_arr[1, i] = 1e9
    else:
        neutral_profile_arr[1, i] = 1e10 * np.exp(-1. * (i - 1.))

result = srs.models.atm.forward(timestamp,
                                latitude,
                                longitude,
                                output,
                                maxwellian_energy_flux=0,
                                gaussian_energy_flux=10,
                                gaussian_peak_energy=50000,
                                gaussian_spectral_width=100,
                                custom_neutral_profile=neutral_profile_arr,
                                d_region=True)

print()
print(result)
print()
result.pretty_print()
print()
