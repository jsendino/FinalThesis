from abc import ABCMeta, abstractmethod
from Appliances.appliance import Appliance


class ApplianceType4(Appliance):
    """
    Class that serves as base for all appliances. Utility() is declared abstract so that each appliance has to
    re-define a proper utility method
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def utility(self, demand, average_demand, household, b, c):
        """
        Computes the utility that one appliance provides to each owner

        :param demand:
        :param average_demand:
        :param household:
        :return: 0
        """
        return 0

