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
"""
Functions for interacting with UCalgary Space Remote Sensing models. One such 
model is the TREx Auroral Transport Model (ATM).
"""

from .atm import ATMManager


class ModelsManager:
    """
    The ModelsManager object is initialized within every PyUCalgarySRS object. It acts 
    as a way to access the submodules and carry over configuration information in the 
    super class.
    """

    def __init__(self, srs_obj):
        self.__srs_obj = srs_obj
        self.__atm = ATMManager(self.__srs_obj)

    @property
    def atm(self):
        """
        Access to the `atm` submodule from within a PyUCalgarySRS object.
        """
        return self.__atm
