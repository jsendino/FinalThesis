#!/usr/bin/env python
"""
Module with parameters to characterize an entertainment type appliance. Also contains the function to obtain its utility
"""
import numpy as np
import scipy

from Appliances.appliance_type_3 import ApplianceType3
from constants import Constants

__author__ = 'jorge'


class Entertainment(ApplianceType3):
    """
    Class that represent one entertainment appliance
    """

    def __init__(self):
        """
        Initialize object with common values (same for all houses)
        """
        # Both expressed in watt/hour
        self.demand_requirement_range = np.array(((1200, 3500),
                                                 (500, 2000)))
        self.demand_range = (0, 400)

        self.working_hours = np.array((np.arange(12, 24),
                                       np.arange(18, 24)))

    def utility(self, demand, average_demand, household, b, c):
        """
        Computes the utility of one entertainment appliance

        :param demand: array with energy demand of appliance in working hours
        :param average_demand: average demand in working hours
        :param household: index in houses array
        :return: array with same elements as hours in which appliance is turned on
        """
        if average_demand == 0:
            return 0
        else:
            return c[household, 4] - \
                   (b[household, 4] +
                    demand / average_demand) ** -1.5
