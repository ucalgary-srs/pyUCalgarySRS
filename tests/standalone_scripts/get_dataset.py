import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

dataset = srs.data.get_dataset("THEMIS_ASI_RAW")

print()
print(dataset)
print()

dataset.pretty_print()
print()
