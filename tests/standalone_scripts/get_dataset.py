import pyucalgarysrs
import pprint

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

dataset = srs.data.get_dataset("TREX_RGB_RAW_NOMINAL")

print()
print(dataset)
print()

print("Example record in dict format:\n------------------------------\n")
pprint.pprint(dataset.__dict__)
