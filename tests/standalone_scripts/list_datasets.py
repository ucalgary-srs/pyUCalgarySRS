import pyucalgarysrs
import pprint

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

datasets = srs.data.list_datasets()

print("\nFound %d datasets" % (len(datasets)))

print("Example record in dict format:\n------------------------------\n")
pprint.pprint(datasets[0].__dict__)
