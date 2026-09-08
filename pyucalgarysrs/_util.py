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

import warnings
import multiprocessing


def show_warning(message: str, stacklevel: int = 1) -> None:
    """
    This is a helper method for within the library to ensure warnings are displayed. Jupyter notebooks
    within VSCode suppress warnings by default, so this way ensures that they are shown.

    NOTE: This is a private method only meant for use within the library.
    """
    warnings.simplefilter("always", UserWarning)
    warnings.warn(message, UserWarning, stacklevel=stacklevel)
    warnings.resetwarnings()


def get_mp_context():
    """
    This is a helper method for within the library to determine the multiprocessing context to
    use for process pools.

    We prefer 'forkserver' where it is available (Linux, macOS). Forking a multi-threaded process
    can deadlock the child, and Python 3.12+ emits a DeprecationWarning for it. Python 3.14 switches
    the Linux default to 'forkserver', so this just adopts that behaviour early. On platforms without
    it (Windows), the default context is used, which is 'spawn'.

    NOTE: This is a private method only meant for use within the library.
    """
    if ("forkserver" in multiprocessing.get_all_start_methods()):
        ctx = multiprocessing.get_context("forkserver")

        # preload this library in the forkserver process instead of the default of '__main__'
        #
        # NOTE: the forkserver process itself imports the '__main__' module of the calling program
        # by default, which fails for programs that have no importable main module (ex. a script
        # piped in on stdin). All the child processes need is this library, so we preload that
        # instead. Note that this does not change the requirement that callers using n_parallel
        # greater than 1 do so from within an `if __name__ == "__main__":` block, since each child
        # process imports the calling script when it starts up.
        ctx.set_forkserver_preload(["pyucalgarysrs"])

        return ctx
    return multiprocessing.get_context()  # pragma: nocover-ok
