import datetime
import pyucalgarysrs
import numpy as np

srs = pyucalgarysrs.PyUCalgarySRS()

output = pyucalgarysrs.ATMForwardOutputFlags()
output.enable_only_height_integrated_rayleighs()
timestamp = datetime.datetime(2021, 11, 4, 6, 0, 0)
latitude = 53.1
longitude = -107.7

num_ef = 11
custom_spectrum_arr = np.zeros((2, num_ef), order='F', dtype=np.single)
for i in range(num_ef):
    custom_spectrum_arr[0, i] = 4000. + i * 100.
    custom_spectrum_arr[1, i] = 1e6

result = srs.models.atm.forward(timestamp, latitude, longitude, output, custom_spectrum=custom_spectrum_arr)

print()
print(result)
print()
result.pretty_print()
print()
