# Copyright 2024 University of Calgary
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import pyucalgarysrs


@pytest.mark.data_observatories
@pytest.mark.parametrize("instrument_array", ["themis_asi", "trex_rgb", "trex_nir", "trex_blue", "rego", "trex_spectrograph", "smile_asi"])
def test_get_sites(srs, instrument_array, capsys):
    # get all sites
    observatories = srs.data.list_observatories(instrument_array)
    assert len(observatories) > 0

    # check serialization
    for o in observatories:
        assert isinstance(o, pyucalgarysrs.Observatory) is True

    # check str method
    print_str = str(observatories[0])
    assert print_str != ""

    # check pretty print method
    observatories[0].pretty_print()
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""


@pytest.mark.data_observatories
@pytest.mark.parametrize("test_dict", [
    {
        "instrument_array": "themis_asi",
        "uid": "atha",
        "expected_results": 1
    },
    {
        "instrument_array": "smile_asi",
        "uid": "atha",
        "expected_results": 1
    },
    {
        "instrument_array": "themis_asi",
        "uid": "ATHA",
        "expected_results": 1
    },
    {
        "instrument_array": "rego",
        "uid": "fs",
        "expected_results": 2
    },
    {
        "instrument_array": "themis_asi",
        "uid": "bad",
        "expected_results": 0
    },
])
def test_get_datasets_filter_name(srs, test_dict):
    observatories = srs.data.list_observatories(test_dict["instrument_array"], uid=test_dict["uid"])
    assert len(observatories) == test_dict["expected_results"]
