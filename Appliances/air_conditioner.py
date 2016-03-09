#!/usr/bin/env python
"""
Module with all the air conditioner parameters, an some functions related to it
"""
import numpy as np
from constants import Constants
from Appliances.appliance_type_1 import ApplianceType1
from demand_response import Demand

__author__ = 'jorge'


class AirConditioner(ApplianceType1):
    """
    Class that represents one air conditioner
    """

    def __init__(self):
        """
        Initialize object with common values (same for all houses)
        """
        self.working_hours = np.array((Constants.day_hours,
                                       np.concatenate((np.arange(18, 24), np.arange(0, 8)))))
        # Parameters that describe how the temperature is regulated
        self.alpha = 0.9
        self.beta = np.random.uniform(-0.011, -0.008)
        # Parameters describing temperature preferences
        self.conf_temp_range = (70, 79)
        self.most_conf_temp = np.random.uniform(73, 77)
        # Matrix of inner temperature (one row per house)
        self.temp_in = np.zeros((Constants.num_households, Constants.day_hours.size))
        self.temp_out = np.array(
            [79, 81, 87, 91, 95, 98, 94, 89, 86, 85, 84, 83, 82, 81, 80, 78, 75, 73, 72, 71, 72, 73, 74, 75])

        self.demand_range = (0, 4000)
        # Array of hours in which air is working. One row per type of house
        self.working_hours = np.array((Constants.day_hours,
                                       np.concatenate((np.arange(18, 24), np.arange(0, 8)))))

    def utility(self, household):
        """
        Computes the utility of using the air conditioner

        :param household: index in houses matrix of current air conditioner
        :return:  a bi-dimensional array of the utility for each house and in hour
        """
        return Constants.c[household, 0] - \
               Constants.b[household, 0] * \
               (self.temp_in - self.most_conf_temp) ** 2

    def update_temperature(self, household, hour):
        """
        Updates the temperature of the household based on the hour and the temperatures before that hour

        :param household: index in households matrix of current air conditioner
        :param hour: hour of day in which temperature is updated
        """
        if hour == 0:
            self.temp_in[household, 0] = 25  # Comfort temperature
        else:
            self.temp_in[household, hour] = self.temp_in[household, hour - 1] + \
                                        self.alpha * (self.temp_out[hour] - self.temp_in[household, hour - 1]) + \
                                        self.beta * Demand.demand[household, 0, hour]
