#! /usr/bin/env python
#
# This program will download all the data that we will need
# for running the test suite quickly.

import datetime
import argparse
import pyucalgarysrs


def main():
    # args
    parser = argparse.ArgumentParser(description="Tool for downloading data necessary for running the full test suite")
    parser.add_argument("--api-url", type=str, default="https://api-staging.phys.ucalgary.ca", help="API base url")
    args = parser.parse_args()

    # setup
    d1 = datetime.datetime.now()
    print("[%s] Starting to download necessary data for tests ..." % (datetime.datetime.now()))
    srs = pyucalgarysrs.PyUCalgarySRS(api_base_url=args.api_url)

    # download THEMIS ASI raw data
    srs.data.download("THEMIS_ASI_RAW", datetime.datetime(2023, 1, 1, 6, 0, 0), datetime.datetime(2023, 1, 1, 6, 59, 59))

    # finish
    print("[%s] Completed downloading data in %s" % (datetime.datetime.now(), (datetime.datetime.now() - d1)))


#---------------------
if (__name__ == "__main__"):
    main()
