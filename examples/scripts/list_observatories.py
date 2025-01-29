import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

observatories = srs.data.list_observatories("themis_asi")

print("\nFound %d observatories\n" % (len(observatories)))
for s in observatories:
    print(s)
print()
