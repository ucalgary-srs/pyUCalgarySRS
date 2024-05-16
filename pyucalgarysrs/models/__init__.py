"""
Provides functions for interacting with UCalgary Space Remote Sensing models. One such 
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
