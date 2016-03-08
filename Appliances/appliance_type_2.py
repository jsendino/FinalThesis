from abc import ABCMeta, abstractmethod
from Appliances.appliance import Appliance


class ApplianceType2(Appliance):
    """
    Class that serves as base for all appliances. Utility() is declared abstract so that each appliance has to
    re-define a proper utility method
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def utility(self, total_demand, household):
        """
        Computes the utility that one appliance provides to each owner

        :param total_demand:
        :param household:
        :return: 0
        """
        return 0

