import pyucalgarysrs
import pprint

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

datasets = srs.data.list_datasets(name="REGO_CALIBRATION")

print("Found %d datasets matching the name filter\n------------------------------\n" % (len(datasets)))
pprint.pprint(datasets)

print("\nExample record in dict format:\n------------------------------\n")
pprint.pprint(datasets[0].__dict__)

print("\npretty_print() method output:\n------------------------------\n")
datasets[0].pretty_print()
