from abc import ABCMeta, abstractmethod
from Appliances.appliance import Appliance


class ApplianceType1(Appliance):
    """
    Class that serves as base for all appliances. Utility() is declared abstract so that each appliance has to
    re-define a proper utility method
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def utility(self, house):
        """
        Computes the utility that one appliance provides to each owner
        :param house:
        :return: 0
        """
        return 0

