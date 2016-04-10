#!/usr/bin/env python
import numpy as np

from constants import Constants
from Appliances.appliance_type_2 import ApplianceType2

__author__ = 'jorge'


class Phev(ApplianceType2):
    """
    Class with parameters to define a PHEV and the function to obtain its utility.
    """

    def __init__(self):
        """
        Initialize object with common values (same for all houses)
        """
        self.demand_range = (0, 2000)  # Minimum and maximum charging rates (in watts)
        self.demand_requirement_range = np.tile((np.random.uniform(4800, 5100), np.random.uniform(5500, 6000)), [2, 1])

        self.working_hours = np.array((np.concatenate((np.arange(18, 24), np.arange(0, 8))),
                                       np.concatenate((np.arange(18, 24), np.arange(0, 8)))))  # Same for all households

    def utility(self, total_demand, household):
        """
        Compute the total demand of PHEV (used only in working hours)
        for each costumer (one computation per house).

        :param total_demand: total demand of customer "household" of current day
        :param household: index of the owner of the washer
        :return: integer with utility
        """
        return Constants.b[household, 1] * total_demand + Constants.c[household, 1]
