import pyucalgarysrs
import pprint

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

dataset = srs.data.get_dataset("THEMIS_ASI_RAW")

print()
print(dataset)
print()

print("Example record in dict format:\n------------------------------\n")
pprint.pprint(dataset.__dict__)
