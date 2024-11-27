#! /usr/bin/env python

import pprint
import datetime
import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()

# get several days of GILL K2 data
#start_dt = datetime.datetime(2005, 6, 4, 0, 0, 0)
#end_dt = datetime.datetime(2005, 6, 4, 23, 59, 59)
#end_dt = datetime.datetime(2010, 1, 5, 23, 59, 59)
start_dt = datetime.datetime(1995, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(1995, 1, 5, 23, 59, 59)
res = srs.data.download("NORSTAR_RIOMETER_K2_TXT", start_dt, end_dt, site_uid="gill", progress_bar_disable=True, n_parallel=5)

# read the data
data = srs.data.read(res.dataset, res.filenames, n_parallel=2)

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
