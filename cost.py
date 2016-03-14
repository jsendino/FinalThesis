#!/usr/bin/env python
from constants import Constants
import numpy as np
from demand_response import Demand
from battery import Battery
import matplotlib.pyplot as plt

__author__ = 'jorge'


class Cost:
    """
    Class with functions to compute total cost of the system
    """
    # Modifier accounting for the magnitude of a real population to multiply total energy demand
    magnitude = 5 * 10 ** 4
    #magnitude = 1 * 10 ** 4

    # Modifier that indicates the units in every step of piecewise function
    M = 1 * 10 ** 6

    # Precision when checking whether the algorithm has converged
    epsilon = 10 ** -3

    # Price increment when re-selling battery
    delta = 0.1

    # Number of samples to compare when checking if price has converged
    range = 1

    # a = (.55 * 100, .8255 * 100)
    # b = (0.70, 1.05)
    # c = (.50 / 100, .75 / 100)

    a = (.1983 * 10, .5285 * 100, .2668 * 1000)
    b = (-.3114 / 10, -.6348, -.2338 * 10)
    c = (.1049 / 100, .2785 / 100, .5935 / 100)

    piecewise_ranges = np.asarray([(0 * M, 138 * M),
                                   (138 * M, 200 * M),
                                   (200 * M, float('inf'))])

    @classmethod
    def total_cost(cls, total_demand):
        """
        Function that given a distribution of demand in different houses computes the cost that will bring to provide
        that amount of  during all day

        :param total_demand: array with the aggregated demand of all houses in every hour of the day
        :rtype : Array with same elements as hours in day
        :return: Each cell of the array is the cost in that hour
        """

        cost = np.zeros(Constants.day_hours.size)
        for t in range(0, Constants.day_hours.size):
            conditions_values = np.array([np.logical_and(cls.piecewise_ranges[i][0] <= total_demand[t],
                                                         total_demand[t] < cls.piecewise_ranges[i][1])
                                          for i in range(0, len(cls.piecewise_ranges))])
            function_values = np.array([np.polynomial.polynomial.polyval(total_demand[t], [cls.a[i], cls.b[i], cls.c[i]])
                                        for i in range(0, len(cls.piecewise_ranges))])

            cost[t] = function_values[conditions_values]

        return cost

    @classmethod
    def compute_price(cls, total_demand):

        """
        Function that computes the energy price as the marginal cost of energy (the derivative of the
        cost function calculated over the aggregated demand)

        :param total_demand: array with the aggregated demand of all houses in every hour of the day
        :return: array with the energy price during all day
        """
        price = np.zeros(Constants.day_hours.size)
        for t in range(0, Constants.day_hours.size):
            conditions_values = np.array([np.logical_and(cls.piecewise_ranges[i][0] <= total_demand[t],
                                                         total_demand[t] < cls.piecewise_ranges[i][1])
                                          for i in range(0, len(cls.piecewise_ranges))])
            function_values = np.array([np.polyval([0, 2 * cls.c[i], cls.b[i]], total_demand[t])
                                        for i in range(0, len(cls.piecewise_ranges))])

            price[t] = function_values[conditions_values]

        return price

    @classmethod
    def energy_price(cls, appliances):
        """
        Function that computes the price of a kw of energy during all day.

        It runs an iterative algorithm in which every house adapt its initial demand
        to the cost until equilibrium is reached.

        :rtype : Bi-dimensional array with same rows as iterations ran and same columns as hours in day.
        :return : Each cell of the returned array is the price in that combination of iteration and hour
        """
        price = np.zeros((Constants.max_num_iterations, Constants.day_hours.size))
        house_price = np.zeros(price.shape)
        i = 0
        while i < Constants.max_num_iterations - 1:
            # Compute aggregated demand during all day
            total_demand = np.sum(Demand.total_house_demand[i], 0) * Constants.num_blocks * cls.magnitude
            # Price is set to the marginal cost
            price[i] = cls.compute_price(total_demand) / (cls.magnitude * Constants.num_blocks * Constants.num_households)
            for j in range(0, Constants.num_households):
                Battery.adapt_charge_rate(j, i, price[i])
                Demand.adapt_demand(j, i, price[i], appliances)
                #Demand.use_battery(j, i)
            house_price[i] = Battery.share_battery(i, price[i])
            # Detect if algorithm has converged. If so, stop iterations
            if i > cls.range and \
                    np.logical_and(price[i, :] * (1 - Cost.epsilon) <= price[range(i - Cost.range, i+1)],
                                   price[range(i - Cost.range, i+1)] <= price[i, :] * (1 + Cost.epsilon)).all():
                cls.final_iteration_number = i + 1
                break
            i += 1
        Constants.final_iteration_number = i + 1  # i + 1 cause counting iteration 0
        return price, house_price

    @classmethod
    def plot_price(cls, price):
        """
        Plots the evolution of the average day price along all iterations
        :param price: Bi-dimensional array with same rows as iterations and columns as hours in day
        """
        average_price = np.nanmean(price[range(0, Constants.final_iteration_number)], 1)

        x = np.arange(0, Constants.final_iteration_number)

        ax = plt.gca()
        ax.set_ylim(0, max(average_price))
        ax.set_xticks(np.arange(0, Constants.final_iteration_number, Constants.max_num_iterations / 100) + 1)
        ax.set_yticks(np.arange(min(average_price), max(average_price)+1, max(average_price)/10))

        plt.plot(x+1, average_price)
        plt.title("Daily average price evolution")
        plt.xlabel("Iteration number")
        plt.ylabel("Average price")
        plt.show()
