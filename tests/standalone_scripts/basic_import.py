import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()

print(srs)

srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

print(srs)

srs.download_output_root_path = "/dev/shm/pyucalgarysrs_data"

print(srs)

print(srs.in_jupyter_notebook)
