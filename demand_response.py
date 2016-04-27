#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import scipy

from battery import Battery
from constants import Constants

__author__ = 'jorge'


class Demand:
    """
    Class with attribute to model the demand of the scenario
    """

    def __init__(self, num_households, num_appliances):
        # demand is a three-dimensional having the energy demand of each appliance for each household during all day
        # (8 household, 6 appliance, 24 hours)
        self.demand = np.ones((num_households,
                               num_appliances,
                               Constants.day_hours.size))

        # total_house_demand is an array containing the total demand of the scenario for each house and each hour
        # in each iteration (used when calculating the price at the beginning of the day)
        self.total_house_demand = np.zeros((Constants.max_num_iterations,
                                            num_households,
                                            Constants.day_hours.size))

        # final demand of the scenario in each hour
        self.q_per_hour = np.zeros(24)

        # gamma is the constant representing the degree of responsiveness of the costumer to a change in demand price
        self.gamma = 0.1

        self.initialize_demand(num_households, num_appliances)

    @staticmethod
    def get_initial_demand(num_households, num_appliances):
        demand = np.reshape(np.loadtxt("initial_demand.txt"),
                            [num_households, num_appliances, Constants.day_hours.size])

        return demand * 2

    def initialize_demand(self, num_households, num_appliances):
        self.demand = Demand.get_initial_demand(num_households,
                                                num_appliances)  # * (1 + abs(np.random.normal(scale=1, size=self.demand.shape)))
        # Total demand at iteration 0
        self.total_house_demand[0] = np.sum(self.demand, axis=1)

    def increment_total_demand(self, iteration, household, hour, charge):
        if charge == 0:
            self.total_house_demand[iteration, household, hour] = 0
        else:
            self.total_house_demand[iteration, household, hour] += charge

    def total_demand_per_hour(self):
        self.q_per_hour = np.sum(self.total_house_demand[Constants.final_iteration_number - 1, :, :], 0)

    def average_demand(self, house, appliance, working_hours):
        if np.sum(working_hours) == 0:
            return 0
        else:
            return sum(self.demand[house, appliance, working_hours]) / np.sum(working_hours)

    def adapt_demand(self, household, household_type, iteration, price, appliances):
        """
        Function that runs a determined household to adapt its demand to current iteration price

        :param household: household that runs the adaption
        :param iteration: iteration of the distributed algorithm in which the update is taking place
        :param price: price of the energy in during present day that the distribution company sends to clients
                      in current iteration
        """
        for i in range(0, appliances.size):
            working_hours = np.in1d(Constants.day_hours,
                                    appliances[i].working_hours[household_type])
            demand_requirement_range = appliances[i].demand_requirement_range[household_type]
            demand_range = appliances[i].demand_range

            utility_derivative = self.get_utility_derivative(appliances, household, i, working_hours)

            if np.any(np.isnan(utility_derivative)):
                utility_derivative[np.isnan(utility_derivative)] = np.nanmean(utility_derivative)

            # Compute demand based on new price
            self.demand[household, i, working_hours] += self.gamma * (utility_derivative - price[working_hours])
            # If computed demand exceeds conditions, adapt it
            exceeded_demand = self.demand[household, i] > max(demand_range)
            self.demand[household, i, np.logical_and(exceeded_demand,
                                                     working_hours)] = max(demand_range)
            insufficient_demand = self.demand[household, i] < min(demand_range)
            self.demand[household, i, np.logical_and(insufficient_demand,
                                                     working_hours)] = min(demand_range)
            # If demand within all day is not in demand requirement range, adapt it
            sum_demand = np.sum(self.demand[household, i, working_hours])
            if sum_demand > max(demand_requirement_range):
                self.demand[household, i, working_hours] -= \
                    (sum_demand - max(demand_requirement_range)) / sum(working_hours)
            elif sum_demand < min(demand_requirement_range):
                self.demand[household, i, working_hours] += \
                    (min(demand_requirement_range) - sum_demand) / sum(working_hours)

        # Compute total demand of current house
        self.total_house_demand[iteration + 1, household] = np.sum(self.demand[household], axis=0)

    def get_utility_derivative(self, appliances, household, i, working_hours):
        base_class = appliances[i]._base_class
        if base_class == "ApplianceType1":
            x = self.demand[household, i]
            arg = (household, working_hours,)
        elif base_class == "ApplianceType2":
            x = np.sum(self.demand[household, i, working_hours])
            arg = (household,)
        else:
            average = self.average_demand(household, i, working_hours)
            x = self.demand[household, i, working_hours]
            arg = (average, household,)
        utility_derivative = scipy.misc.derivative(appliances[i].utility,
                                                   x,
                                                   dx=0.01,
                                                   args=arg)
        return utility_derivative

    def plot_demand(self, num_households, household_type, num_appliances, charge_rate):
        """
        Function that plots the demand of each appliance during all day for each household type
        """
        # Set vector of colors
        colors = ('b', 'm', 'c', 'k', 'g')
        # Set vector of types
        types = ('I',) * (num_households - np.count_nonzero(household_type)) + \
                ("II",) * np.count_nonzero(household_type)

        legends = ["Air conditioner", "PHEV", "Washer", "Lights", "TV"]
        locs = "upper right" * (num_households - np.count_nonzero(household_type)) + \
               "upper left" * np.count_nonzero(household_type)

        index_type1 = np.nonzero(household_type)
        for household in (0, index_type1[0][0]):
            # Set x axis values
            ax = plt.gca()
            x = range(0, 24)
            ax.set_xticks(x)
            # Set x axis labels
            labels = [str(hour) for hour in Constants.day_hours]
            ax.set_xticklabels(labels)
            # Plot demand of every appliance
            for i in range(0, num_appliances):
                plt.step(x, self.demand[household, i, :], color=colors[i], where='post', label=legends[i])
            plt.step(x, charge_rate[household], color="r", where="post", linestyle="dashed", label="Battery")
            plt.step(x, self.total_house_demand[Constants.final_iteration_number - 1, household], color='r',
                     where='post', label="Total")
            plt.title("Demand of a type " + str(types[household]) + " household")
            plt.xlim(min(x), max(x) + 1)
            plt.legend(loc=locs[i], labelspacing=0.1)
            plt.show()

    @staticmethod
    def plot_total_demand():
        initial_total_demand = np.sum(np.sum(Demand.get_initial_demand(), 1), 0)
        final_flat1 = np.loadtxt("priceno.txt")
        final_flat2 = np.loadtxt("pricesi.txt")

        # Set x axis values
        ax = plt.gca()
        x = range(0, 24)
        ax.set_xticks(x)
        # Set x axis labels
        labels = [str(hour) for hour in Constants.day_hours]
        ax.set_xticklabels(labels)
        # plt.step(x, initial_total_demand, color='b', where='post', label="No demand response")
        plt.step(x, final_flat1, color='m', where='post', label="No battery sharing")
        plt.step(x, final_flat2, color='g', where='post', label="Battery is shared")
        plt.title("Electricity demand response under different battery schemes")
        plt.xlim(min(x), max(x) + 1)
        plt.legend(loc='upper right')
        plt.show()
