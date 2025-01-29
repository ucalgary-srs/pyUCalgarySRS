import pyucalgarysrs
import pprint

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

observatories = srs.data.list_observatories("themis_asi", uid="fs")

print("\nFound %d observatories matching the uid filter\n" % (len(observatories)))
for o in observatories:
    print(o)
print()

print("\nExample record in dict format:\n------------------------------\n")
pprint.pprint(observatories[0].__dict__)

print("\npretty_print() method output:\n------------------------------\n")
observatories[0].pretty_print()
