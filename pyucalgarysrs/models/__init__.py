from ._atm import ATMManager


class ModelsManager:

    def __init__(self, srs_obj):
        self.__srs_obj = srs_obj
        self.atm = ATMManager(self.__srs_obj)
