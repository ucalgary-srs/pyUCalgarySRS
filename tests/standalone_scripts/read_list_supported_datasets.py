import pyucalgarysrs

# init
srs = pyucalgarysrs.PyUCalgarySRS()

# get list
datasets = srs.data.list_supported_read_datasets()

print()
print(datasets)
