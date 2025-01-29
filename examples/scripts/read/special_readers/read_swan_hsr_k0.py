import pprint
import datetime
import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()

# get several days of K0 data
start_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(2024, 1, 10, 23, 59, 59)
res = srs.data.download("SWAN_HSR_K0_H5", start_dt, end_dt, site_uid="mean", progress_bar_disable=True, n_parallel=5)

# read the data
data = srs.data.readers.read_swan_hsr(res.filenames, n_parallel=2, dataset=res.dataset)

print(data)

data.pretty_print()

print()

data.data[0].pretty_print()

print()

print(data.data[0].timestamp[0:10])

print()

pprint.pprint(data.timestamp)

print()

pprint.pprint(data.metadata[0])
