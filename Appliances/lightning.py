#!/usr/bin/env python
"""
Module with all the parameters to define a lightning type appliance and the function to obtain its utility
"""
from constants import Constants
from demand_response import Demand
from appliance_type_4 import ApplianceType4

import numpy as np

__author__ = 'jorge'


class Lightning(ApplianceType4):
    """
    Class that represent the house lightning
    """

    def __init__(self):
        """
        Initialize object with common values (same for all houses)
        """
        self.demand_range = (200, 800)

        self.working_hours = np.array((np.arange(18, 24),
                                       np.arange(18, 24)))

    def utility(self, demand, average_demand, household):
        """
        Computes the utility of the lightning

        :param demand: array with energy demand of appliance in working hours
        :param average_demand: average demand in working hours
        :param household: index in houses array
        :return: array with same elements as hours in which appliance is turned on
        """
        if average_demand == 0:
            return 0
        else:
            return Constants.c[household, 3] - \
                   (Constants.b[household, 3] +
                    demand / average_demand) ** -1.5
