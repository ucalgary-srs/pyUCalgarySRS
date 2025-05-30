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

import shutil
import os
import random
import string
import pytest
import platform
import pyucalgarysrs
import warnings
from pathlib import Path


@pytest.mark.top_level
def test_top_level_class_instantiation_noparams():
    # instantiate
    srs = pyucalgarysrs.PyUCalgarySRS()

    # initialize paths
    srs.initialize_paths()
    assert os.path.exists(srs.download_output_root_path)
    assert os.path.exists(srs.read_tar_temp_path)

    # change download root path
    new_path = str("%s/pyucalgarysrs_data_download_testing_%s" % (Path.home(), ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))))
    srs.download_output_root_path = new_path
    assert srs.download_output_root_path == new_path
    srs.initialize_paths()
    assert os.path.exists(new_path)
    shutil.rmtree(new_path, ignore_errors=True)

    # change tar temp path
    new_path = str("%s/pyucalgarysrs_data_tar_testing%s" % (Path.home(), ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))))
    srs.read_tar_temp_path = new_path
    assert srs.read_tar_temp_path == new_path
    srs.initialize_paths()
    assert os.path.exists(new_path)
    shutil.rmtree(new_path, ignore_errors=True)

    # check __str__ and __repr__ for PyUCalgarySRS type
    print_str = str(srs)
    assert print_str != ""
    assert isinstance(str(srs), str) is True
    assert isinstance(repr(srs), str) is True


@pytest.mark.top_level
def test_top_level_class_instantiation_usingparams():
    # instantiate object
    testing_url = "https://testing-url.com"
    testing_download_path = str("%s/pyucalgarysrs_data_download_testing_%s" %
                                (Path.home(), ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))))
    testing_read_path = str("%s/pyucalgarysrs_data_tar_testing%s" %
                            (Path.home(), ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))))
    testing_api_key = "abcd1234"
    testing_api_timeout = 5
    testing_api_headers = {"some_key": "some value"}
    srs = pyucalgarysrs.PyUCalgarySRS(
        api_base_url=testing_url,
        download_output_root_path=testing_download_path,
        read_tar_temp_path=testing_read_path,
        api_key=testing_api_key,
        api_timeout=testing_api_timeout,
        api_headers=testing_api_headers,
    )
    assert srs.download_output_root_path == testing_download_path
    assert srs.read_tar_temp_path == testing_read_path
    assert srs.api_base_url == testing_url
    assert srs.api_timeout == testing_api_timeout
    assert srs.api_headers == testing_api_headers
    assert srs.api_key == testing_api_key
    assert os.path.exists(testing_download_path) is False
    assert os.path.exists(testing_read_path) is False


@pytest.mark.top_level
def test_bad_paths_noparams(srs):
    # test bad paths
    #
    # NOTE: we only do this check on Linux since I don't know a bad
    # path to check on Mac. Good enough for now.
    if (platform.system() == "Linux"):
        new_path = "/dev/bad_path"
        with pytest.raises(pyucalgarysrs.SRSInitializationError) as e_info:
            srs.download_output_root_path = new_path
            srs.initialize_paths()
        assert "Error during output path creation" in str(e_info)
        with pytest.raises(pyucalgarysrs.SRSInitializationError) as e_info:
            srs.read_tar_temp_path = new_path
            srs.initialize_paths()
        assert "Error during output path creation" in str(e_info)


@pytest.mark.top_level
def test_api_base_url(srs):
    # set flag
    srs.api_base_url = "https://something"
    assert srs.api_base_url == "https://something"
    srs.api_base_url = None
    assert srs.api_base_url != "https://something"

    # check that trailing slash is removed
    srs.api_base_url = "https://something/"
    assert srs.api_base_url == "https://something"

    # check invalid URL
    with pytest.raises(pyucalgarysrs.SRSInitializationError) as e_info:
        srs.api_base_url = "something invalid"
    assert "API base URL is an invalid URL" in str(e_info)


@pytest.mark.top_level
def test_api_headers(srs):
    # set flag
    default_headers = srs.api_headers
    srs.api_headers = {"some": "thing"}
    assert "some" in srs.api_headers and srs.api_headers["some"] == "thing"
    srs.api_headers = None
    assert srs.api_headers == default_headers

    # set user-agent header as if we were pyaurorax
    pyaurorax_useragent = "python-pyaurorax/someversion"
    srs.api_headers = {"user-agent": pyaurorax_useragent}
    assert srs.api_headers["user-agent"] == pyaurorax_useragent

    # check warning
    with warnings.catch_warnings(record=True) as w:
        srs.api_headers = {"user-agent": "some other value"}
    assert len(w) == 1
    assert issubclass(w[-1].category, UserWarning)
    assert "Cannot override default" in str(w[-1].message)


@pytest.mark.top_level
def test_api_timeout(srs):
    # set flag
    default_timeout = srs.api_timeout
    srs.api_timeout = 5
    assert srs.api_timeout == 5
    srs.api_timeout = None
    assert srs.api_timeout == default_timeout


@pytest.mark.top_level
def test_progress_bar_backend(srs):
    # save default for later
    progress_bar_backend = srs.progress_bar_backend

    # set flag (standard)
    srs.progress_bar_backend = "standard"
    assert srs.progress_bar_backend == "standard"

    # set flag (notebook)
    srs.progress_bar_backend = "notebook"
    assert srs.progress_bar_backend == "notebook"

    # set flag (auto)
    srs.progress_bar_backend = "auto"
    assert srs.progress_bar_backend == "auto"

    # set flag (back to default)
    srs.progress_bar_backend = None
    assert srs.progress_bar_backend == progress_bar_backend

    # check invalid value
    with pytest.raises(pyucalgarysrs.SRSInitializationError) as e_info:
        srs.progress_bar_backend = "something invalid"
    assert "Invalid progress bar backend" in str(e_info)


@pytest.mark.top_level
def test_purge_download_path(srs):
    # set up object
    #
    # NOTE: we set the path to something with a random string in it
    # so that our github actions for linux/mac/windows, which fire off
    # simultaneously on the same machine, work without stepping on the
    # toes of each other.
    new_path = str("%s/pyucalgarysrs_data_purge_download_testing_%s" %
                   (Path.home(), ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))))
    srs.download_output_root_path = new_path
    assert srs.download_output_root_path == new_path
    assert os.path.exists(srs.download_output_root_path) is False
    srs.initialize_paths()
    assert os.path.exists(srs.download_output_root_path) is True

    # create some dummy files and folders
    os.makedirs("%s/testing1" % (srs.download_output_root_path), exist_ok=True)
    os.makedirs("%s/testing2" % (srs.download_output_root_path), exist_ok=True)
    os.makedirs("%s/testing2/testing3" % (srs.download_output_root_path), exist_ok=True)
    Path("%s/testing.txt" % (srs.download_output_root_path)).touch()
    Path("%s/testing1/testing.txt" % (srs.download_output_root_path)).touch()

    # check purge function
    srs.purge_download_output_root_path()
    assert len(os.listdir(srs.download_output_root_path)) <= 1  # only the tar_temp_dir should be there

    # cleanup
    shutil.rmtree(srs.download_output_root_path, ignore_errors=True)


@pytest.mark.top_level
def test_purge_tar_temp_path(srs):
    # set up object
    new_path = str("%s/pyucalgarysrs_data_purge_tartemp_testing_%s" %
                   (Path.home(), ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))))
    srs.read_tar_temp_path = new_path
    assert srs.read_tar_temp_path == new_path
    assert os.path.exists(srs.read_tar_temp_path) is False
    srs.initialize_paths()
    assert os.path.exists(srs.read_tar_temp_path) is True

    # create some dummy files and folders
    os.makedirs("%s/testing1" % (srs.read_tar_temp_path), exist_ok=True)
    os.makedirs("%s/testing2" % (srs.read_tar_temp_path), exist_ok=True)
    os.makedirs("%s/testing2/testing3" % (srs.read_tar_temp_path), exist_ok=True)
    Path("%s/testing.txt" % (srs.read_tar_temp_path)).touch()
    Path("%s/testing1/testing.txt" % (srs.read_tar_temp_path)).touch()

    # check purge function
    srs.purge_read_tar_temp_path()
    assert len(os.listdir(srs.read_tar_temp_path)) == 0

    # cleanup
    shutil.rmtree(srs.read_tar_temp_path, ignore_errors=True)


@pytest.mark.top_level
def test_show_data_usage(srs, capsys):
    # check default params
    print(srs.show_data_usage())
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check return_dict=True
    print(srs.show_data_usage(return_dict=True))
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""

    # check order being name
    print(srs.show_data_usage(order="name"))
    captured_stdout = capsys.readouterr().out
    assert captured_stdout != ""
