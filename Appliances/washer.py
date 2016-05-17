#!/usr/bin/env python
"""
Module with necessary parameters to define a washer type appliance and the function to obtain its utility.
"""
import numpy as np

from Appliances.appliance_type_2 import ApplianceType2
from constants import Constants

__author__ = 'jorge'


class Washer(ApplianceType2):
    """
    Module with parameters to define a washer and the function to obtain its utility.
    """

    def __init__(self):
        """
        Initialize object with common values (same for all houses)
        """
        self.demand_requirement_range = np.tile((np.random.uniform(1400, 1600), np.random.uniform(2000, 2500)), [2, 1])
        self.demand_range = (0, 1500)

        self.working_hours = np.array((Constants.day_hours,
                                       np.concatenate((np.arange(18, 24), np.arange(0, 8)))))

    def utility(self, total_demand, household, b, c):
        """
        Compute the total demand of washer (used only in working hours)
        for each costumer (one computation per house)

        :param total_demand: total demand of customer "household" of current day
        :param household: index of the owner of the washer
        :return: integer with utility
        """
        return total_demand + c[household, 2]
