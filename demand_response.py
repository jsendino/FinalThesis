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
    # demand is a three-dimensional having the energy demand of each appliance for each household during all day
    # (8 household, 6 appliance, 24 hours)
    demand = np.ones((Constants.num_households,
                      Constants.num_appliances,
                      Constants.day_hours.size))

    # total_house_demand is an array containing the total demand of the scenario for each house and each hour
    # in each iteration (used when calculating the price at the beginning of the day)
    total_house_demand = np.zeros((Constants.max_num_iterations,
                                   Constants.num_households,
                                   Constants.day_hours.size))

    # final demand of the scenario in each hour
    q_per_hour = np.zeros(24)

    # gamma is the constant representing the degree of responsiveness of the costumer to a change in demand price
    gamma = 0.1

    @staticmethod
    def get_initial_demand():
        demand = np.ones((Constants.num_households,
                          Constants.num_appliances,
                          Constants.day_hours.size))
        # Demands for the type one appliance
        demand[range(0, int(Constants.num_households / 2)), 0, :] = np.concatenate(
            [np.array((0, 400, 800, 1200, 1400, 1680, 1350, 1000, 830, 800, 575, 500, 400, 200)),
             np.zeros(10)])
        demand[range(int(Constants.num_households / 2), Constants.num_households), 0, :] = np.concatenate([np.zeros(10),
                                                                                                           np.array((
                                                                                                                    800,
                                                                                                                    600,
                                                                                                                    500,
                                                                                                                    300)),
                                                                                                           np.zeros(
                                                                                                               10)])

        # Demands for type 2 app
        demand[range(0, int(Constants.num_households / 2)), 1, :] = np.concatenate([np.zeros(10),
                                                                                    np.array((50, 100, 250, 250, 450,
                                                                                              200, 650, 400, 420, 650,
                                                                                              600,
                                                                                              600, 625, 400))])
        demand[range(int(Constants.num_households / 2), Constants.num_households), 1, :] = np.concatenate([np.zeros(14),
                                                                                                           np.array((
                                                                                                                    450,
                                                                                                                    200,
                                                                                                                    500,
                                                                                                                    550,
                                                                                                                    750,
                                                                                                                    500,
                                                                                                                    750,
                                                                                                                    760,
                                                                                                                    700,
                                                                                                                    775))])
        # Demands for type 3
        demand[range(0, int(Constants.num_households / 2)), 2, :] = np.concatenate([np.array((800, 600, 450, 300)),
                                                                                    np.zeros(3),
                                                                                    np.array((25, 50, 100)),
                                                                                    np.zeros(14)])
        demand[range(int(Constants.num_households / 2), Constants.num_households), 2, :] = np.concatenate([np.zeros(16),
                                                                                                           np.array((
                                                                                                                    300,
                                                                                                                    350,
                                                                                                                    325,
                                                                                                                    300,
                                                                                                                    400,
                                                                                                                    300,
                                                                                                                    375,
                                                                                                                    350))])
        # Demands for type 4
        demand[range(0, int(Constants.num_households / 2)), 3, :] = np.concatenate([np.zeros(10),
                                                                                    np.array(
                                                                                        (200, 225, 250, 275, 300, 300)),
                                                                                    np.zeros(8)])
        demand[range(int(Constants.num_households / 2), Constants.num_households), 3, :] = np.concatenate([np.zeros(10),
                                                                                                           np.array((
                                                                                                                    200,
                                                                                                                    200,
                                                                                                                    225,
                                                                                                                    300,
                                                                                                                    350,
                                                                                                                    375)),
                                                                                                           np.zeros(8)])
        # Demands for type 5
        demand[range(0, int(Constants.num_households / 2)), 4, :] = np.concatenate([np.zeros(4),
                                                                                    np.array((220, 225, 200, 250, 300,
                                                                                              250, 200, 375, 400, 400,
                                                                                              250,
                                                                                              200)),
                                                                                    np.zeros(8)])
        demand[range(int(Constants.num_households / 2), Constants.num_households), 4, :] = np.concatenate([np.zeros(10),
                                                                                                           np.array((
                                                                                                                    300,
                                                                                                                    375,
                                                                                                                    400,
                                                                                                                    400,
                                                                                                                    225,
                                                                                                                    300)),
                                                                                                           np.zeros(8)])

        return demand

    @classmethod
    def initialize_demand(cls):
        cls.demand = Demand.get_initial_demand() #* (1 + abs(np.random.normal(scale=0.5, size=cls.demand.shape)))

        cls.total_house_demand[0] = np.sum(cls.demand, axis=1)

    @classmethod
    def compute_demand_response(cls, household, appliance_type, hour, is_with_battery):
        """
        Computes the initial energy demand response of each household. This is the demand
        that each household makes without adjusting it to current price

        :param household: integer that indicates the household index
        :param appliance_type:  integer that indicates the appliance type (1-5)
        :param hour: hour of the day in which the demand has to be computed
        :param is_with_battery:  boolean to indicate whether a battery is being used or not
        :return:
        :rtype : integer representing the demand in watts
        """
        return cls.demand[household, appliance_type, hour]

    @classmethod
    def total_demand_per_hour(cls, t):
        Demand.q_per_hour[t] = np.sum(cls.demand[:, :, t])

    @classmethod
    def average_demand(cls, house, appliance, working_hours):
        if np.sum(working_hours) == 0:
            return 0
        else:
            return sum(cls.demand[house, appliance, working_hours]) / np.sum(working_hours)

    @classmethod
    def adapt_demand(cls, household, iteration, price, appliances):
        """
        Function that runs a determined household to adapt its demand to current iteration price

        :param household: household that runs the adaption
        :param iteration: iteration of the distributed algorithm in which the update is taking place
        :param price: price of the energy in during present day that the distribution company sends to clients
                      in current iteration
        """
        for i in range(0, appliances.size):
            working_hours = np.in1d(Constants.day_hours,
                                    appliances[i].working_hours[Constants.household_type[household]])
            demand_range = appliances[i].demand_range
            x = arg = 0

            base_class = appliances[i]._base_class
            if base_class == "ApplianceType1":
                utility_derivative = 0
            else:
                if base_class == "ApplianceType2":
                    x = np.sum(cls.demand[household, i, working_hours])
                    arg = (household, )
                elif base_class == "ApplianceType3" or base_class == "ApplianceType4":
                    average = cls.average_demand(household, i, working_hours)
                    x = cls.demand[household, i, working_hours]
                    arg = (average, household,)

                utility_derivative = scipy.misc.derivative(appliances[i].utility,
                                                           x,
                                                           dx=0.01,
                                                           args=arg)
                # If utility is not iterable, IndexError is raised and that means that is an integer.
                # try:
                #     utility_derivative = utility_derivative[working_hours]
                # except IndexError:
                #     pass

            # Compute demand based on new price
            cls.demand[household, i, working_hours] += cls.gamma * (utility_derivative - price[working_hours])
            # If computed demand exceeds conditions, adapt it
            exceeded_demand = cls.demand[household, i] > max(demand_range)
            cls.demand[household, i, np.logical_and(exceeded_demand,
                                                    working_hours)] = max(demand_range)
            insufficient_demand = cls.demand[household, i] < min(demand_range)
            cls.demand[household, i, np.logical_and(insufficient_demand,
                                                    working_hours)] = min(demand_range)

        # Compute total demand of current house
        cls.total_house_demand[iteration+1, household] = np.sum(cls.demand[household], axis=0)

    @classmethod
    def use_battery(cls, household, iteration):
        """
        Function that takes into account the use of the battery: if it is charging this charge rate will be added, else
        it will be subtracted
        :param household:
        :param iteration:
        """
        cls.total_house_demand[iteration + 1, household] += Battery.charge_rate[household]
        # If demand is lower that 0 in some cases, it means that battery must be stopped in those hours
        null_demand_hours = cls.total_house_demand[iteration + 1, household] < 0
        # Correct battery usage
        Battery.charge_rate[household, null_demand_hours] -= \
            cls.total_house_demand[iteration + 1, household, null_demand_hours]
        Battery.check_battery(household, iteration)
        # Set demand to 0
        cls.total_house_demand[iteration + 1, household, null_demand_hours] = 0

    @staticmethod
    def plot_demand():
        """
        Function that plots the demand of each appliance during all day for each household type
        """
        # Set vector of colors
        colors = ('b', 'm', 'c', 'k', 'g')
        # Set vector of types
        types = ('I',) * (Constants.num_households - np.count_nonzero(Constants.household_type)) + \
                ("II",) * np.count_nonzero(Constants.household_type)

        index_type1 = np.nonzero(Constants.household_type)
        for household in (0, index_type1[0][0]):
            # Set x axis values
            ax = plt.gca()
            x = range(0, 24)
            ax.set_xticks(x)
            # Set x axis labels
            labels = [str(hour) for hour in Constants.day_hours]
            ax.set_xticklabels(labels)
            # Plot demand of every appliance
            for i in range(0, Constants.num_appliances):
                plt.step(x, Demand.demand[household, i, :], color=colors[i], where='post')
                plt.step(x, Battery.charge_rate[household], color="r", where="post", linestyle="dashed")
            plt.step(x, Demand.total_house_demand[Constants.final_iteration_number - 1, household], color='r', where='post')
            plt.title("Demand of a type " + str(types[household]) + " household")
            plt.xlim(min(x), max(x) + 1)
            plt.show()

    @staticmethod
    def plot_total_demand():
        initial_total_demand = np.sum(np.sum(Demand.get_initial_demand(), 1), 0)
        final_flat1 = np.loadtxt("flat1.txt")
        final_flat2 = np.loadtxt("flat2.txt")
        final_bat = np.loadtxt("real-time-bat.txt")
        final_no_bat = np.loadtxt("real-time-nobat.txt")

        # Set x axis values
        ax = plt.gca()
        x = range(0, 24)
        ax.set_xticks(x)
        # Set x axis labels
        labels = [str(hour) for hour in Constants.day_hours]
        ax.set_xticklabels(labels)
        plt.step(x, initial_total_demand, color='b', where='post', label="No demand response")
        plt.step(x, final_flat1, color='m', where='post', label="Flat price scheme I")
        plt.step(x, final_flat2, color='k', where='post', label="Flat price scheme II")
        plt.step(x, final_bat, color='r', where='post', label="Real time; no battery")
        plt.step(x, final_no_bat, color='c', where='post', label="Real time; with battery")
        plt.title("Electricity demand response under different schemes")
        plt.xlim(min(x), max(x) + 1)
        plt.legend(loc='upper left')
        plt.show()

