import pprint
import datetime
import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()

# get the data
start_dt = datetime.datetime(2009, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(2009, 1, 5, 23, 59, 59)
res = srs.data.download("NORSTAR_RIOMETER_K2_TXT", start_dt, end_dt, site_uid="gill", progress_bar_disable=True, n_parallel=5)

# read the data
data = srs.data.readers.read_norstar_riometer(res.filenames, n_parallel=2, dataset=res.dataset)

print(data)

data.pretty_print()

print()

data.data[0].pretty_print()

print()

print(data.data[0].timestamp[0:10])
print(data.data[0].timestamp[-10:])

print()

pprint.pprint(data.timestamp)

print()

pprint.pprint(data.metadata[0])
