import pyucalgarysrs

# init
srs = pyucalgarysrs.PyUCalgarySRS()

# check
print()
print("THEMIS_ASI_RAW supported: %s" % (srs.data.is_read_supported("THEMIS_ASI_RAW")))
print("SOME_BAD_DATASET supported: %s" % (srs.data.is_read_supported("THEMIS_SOME_BAD_DATASETASI_RAW")))
print()
