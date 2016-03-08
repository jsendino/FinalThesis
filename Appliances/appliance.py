from abc import ABCMeta


class Appliance:
    """
    Class that serves as base for all appliances. Utility() is declared abstract so that each appliance has to
    re-define a proper utility method
    """
    __metaclass__ = ABCMeta

    # Demand range of an appliance (min and max demand per hour)
    demand_range = (0, 0)
    # Hours of day in which this appliance is working.
    working_hours = 0

    @property
    def _base_class(self):
        """
        :return: Returns the name of the base class
        """
        return self.__class__.__base__.__name__
