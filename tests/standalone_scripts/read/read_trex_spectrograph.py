import pyucalgarysrs
import datetime
import glob

# init
srs = pyucalgarysrs.PyUCalgarySRS()

# set list of files (we could do this using a glob too)
file_list = sorted(glob.glob("/mnt/ceph/trex/spectrograph/stream0/2023/01/01/rabb*/ut06/*_060[0,1,2]_*.pgm*"))

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data = srs.data.readers.read_trex_spectrograph(file_list, n_parallel=2)

print()
print(data)
