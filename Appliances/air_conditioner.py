#!/usr/bin/env python
"""
Module with all the air conditioner parameters, an some functions related to it
"""
import numpy as np

from Appliances.appliance_type_1 import ApplianceType1
from constants import Constants

__author__ = 'jorge'


class AirConditioner(ApplianceType1):
    """
    Class that represents one air conditioner
    """

    def __init__(self):
        """
        Initialize object with common values (same for all houses)
        """
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
        # Demand rate range (watts per hour)
        self.demand_range = (0, 4000)
        # Total min and max demand for each type of house
        self.demand_requirement_range = np.tile((0, float('inf')), [2, 1])
        # Array of hours in which air is working. One row per type of house
        self.working_hours = np.array((Constants.day_hours,
                                       np.concatenate((np.arange(18, 24), np.arange(0, 8)))))

    def utility(self, demand, household, working_hours):
        """
        Computes the utility of using the air conditioner

        :param demand: demand used by the air conditioner
        :param working_hours: hours in which it is working
        :param household: index in houses matrix of current air conditioner
        :return: an array of the utility of this appliance for house "household" in working_hours
        """
        utility = Constants.c[household, 0] - \
                  Constants.b[household, 0] * \
                  ((self.compute_temp_in(demand, household) - self.most_conf_temp) ** 2)

        return utility[working_hours]

    def compute_temp_in(self, demand, household):
        """
        Updates the temperature of the household based on the hour and the temperatures before that hour

        :param demand: demand used by the air conditioner
        :return: inner temperature
        :param household: index in households matrix of current air conditioner
        """
        for hour in range(Constants.day_hours.size):
            if hour == 0:
                self.temp_in[household, 0] = self.most_conf_temp  # Comfort temperature
            else:
                self.temp_in[household, hour] = (1-self.alpha)**hour * self.temp_in[household, 0]
                for t in range(1, hour+1):
                    self.temp_in[household, hour] += (1-self.alpha) ** (hour-t) * self.alpha * self.temp_out[t] #+ \
                                                     #(1-self.alpha) ** (hour-t) * self.beta * demand[t]

                # if self.temp_in[household, hour] > max(self.conf_temp_range):
                #     self.temp_in[household, hour] = max(self.conf_temp_range)
                # elif self.temp_in[household, hour] < min(self.conf_temp_range):
                #     self.temp_in[household, hour] = min(self.conf_temp_range)

        return self.temp_in[household]