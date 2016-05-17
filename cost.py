#!/usr/bin/env python
from constants import Constants
import numpy as np
from demand_response import Demand
from battery import Battery
import matplotlib.pyplot as plt
import scipy

__author__ = 'jorge'


class Cost:
    """
    Class with functions to compute the cost of the system
    """
    # Array with price at which the energy is re-selled
    house_price = np.zeros((Constants.max_num_iterations, Constants.day_hours.size))

    # Modifier accounting for the magnitude of a real population to multiply total energy demand
    magnitude = 5 * 10 ** 5
    # magnitude = 1 * 10 ** 6

    # Modifier that indicates the units in every step of piecewise function
    M = 1 * 10 ** 6

    # Precision when checking whether the algorithm has converged
    epsilon = 10 ** -3

    # Price increment when resaling battery
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
        conditions_values = np.array([np.logical_and(cls.piecewise_ranges[i][0] <= total_demand,
                                                     total_demand < cls.piecewise_ranges[i][1])
                                      for i in range(0, len(cls.piecewise_ranges))])
        function_values = np.array([np.polyval([cls.c[i], cls.b[i], cls.a[i]], total_demand)
                                    for i in range(0, len(cls.piecewise_ranges))])

        return function_values[conditions_values]

    @classmethod
    def energy_price(cls, total_demand, num_households):

        """
        Function that computes the energy price as the marginal cost of energy (the derivative of the
        cost function calculated over the aggregated demand)

        :param total_demand: integer with the aggregated demand of all houses in one hour or array with aggregated demands
        :return: integer/array with the energy price corresponding to total_demand
        """
        price = np.zeros(total_demand.shape)
        for i in range(len(total_demand)):
            p = scipy.misc.derivative(cls.total_cost, total_demand[i] * cls.magnitude, dx=0.1) / \
               (cls.magnitude * num_households)
            if np.sum(p) == 0:
                p = 0
            price[i] = p
        return price

    @classmethod
    def optimal_demand_response(cls, consumers):
        """
        Function that computes the price of a kw of energy during all day.

        It runs an iterative algorithm in which every house adapt its initial demand
        to the cost until equilibrium is reached.

        :param consumers: Consumer object
        :rtype : Bi-dimensional array with same rows as iterations ran and same columns as hours in day.
        :return : Each cell of the returned array is the price in that combination of iteration and hour
        """
        i = 0
        global_price = np.zeros((Constants.max_num_iterations, Constants.num_producers, Constants.day_hours.size))
        auction_price = np.zeros((Constants.max_num_iterations, Constants.day_hours.size))
        while i < Constants.max_num_iterations - 1:
            # For every producer, get its price based on the demand each one has to serve in the previous iteration
            for prod in range(Constants.num_producers):
                total_demand = consumers.get_demand_by_prod(i, prod, range(Constants.day_hours.size))
                global_price[i, prod] = cls.energy_price(total_demand, consumers.num_total_households)

            auction_price[i] = cls.auction(consumers, global_price[i], i)
            global_price[i, :] = auction_price[i, :]

            # Now prices has been established, each house can decide from which provider be served in next iteration
            # consumers.decide_provider(i, global_price[i])
            # Detect if algorithm has converged. If so, stop iterations
            n_samples_mean = auction_price[range(i - Cost.range, i+1), :]
            if i > cls.range and \
                    np.logical_and(auction_price[i] * (1 - Cost.epsilon) <= n_samples_mean,
                                   n_samples_mean <= auction_price[i] * (1 + Cost.epsilon)).all():
                break
            # For every house, take price for all the day based on the provider decision every house has made
            # and adapt demand
            for j in range(consumers.blocks.size):
                consumers.blocks[j].update_parameters(i, global_price[i], consumers.blocks[j].appliances)

                cls.house_price[i] = consumers.blocks[j].start_battery_market(i, np.mean(global_price[i], 0))
            i += 1
        Constants.final_iteration_number = i + 1  # i + 1 cause counting iteration 0

        return global_price

    @classmethod
    def auction(cls, consumers, costs, iteration):
        sellers = range(Constants.num_producers)
        buyers = range(consumers.num_total_households)

        redemptions = np.zeros((consumers.num_total_households, Constants.day_hours.size))
        for i in range(Constants.day_hours.size):
            redemptions[:, i] = consumers.get_initial_bids(iteration, i)

        transactions = []
        alltrans = []
        gain = 0
        last = np.tile(-999, Constants.day_hours.size)
        for r in range(consumers.num_total_households):
            ask =  np.tile(100.0, Constants.day_hours.size)
            seller = -1
            bid = np.tile(0.0, Constants.day_hours.size)
            buyer = -1
            for r in range(100):
                if np.random.random() < 0.5:
                    i = np.random.choice(sellers)
                    new_ask = np.random.random() * (100 - costs[i]) + costs[i]

                    if np.all(new_ask < ask):
                        ask = new_ask
                        last = ask
                        seller = i
                else:
                    j = np.random.choice(buyers)
                    new_bid = np.random.random() * redemptions[j]
                    if np.all(new_bid > bid):
                        bid = new_bid
                        last = bid
                        buyer = j
                if np.all(ask < bid):
                    # sellers -= {seller}
                    # buyers -= {buyer}
                    # transactions.append(last)
                    # assert last >= costs[seller]
                    # assert last <= redemptions[buyer]
                    # gain += (redemptions[buyer] - costs[seller])
                    break

        return last

    @classmethod
    def plot_price(cls, price):
        """
        Plots the evolution of the average day price along all iterations
        :param price: Bi-dimensional array with same rows as iterations and columns as hours in day
        """
        average_price = np.nanmean(np.nanmean(price[range(0, Constants.final_iteration_number)], 2), 1)

        x = np.arange(0, Constants.final_iteration_number)

        ax = plt.gca()
        ax.set_ylim(0, max(average_price))
        ax.set_xticks(np.arange(0, Constants.final_iteration_number, Constants.max_num_iterations / 100) + 1)
        ax.set_yticks(np.arange(min(average_price), max(average_price) + 1, max(average_price) / 10))

        plt.plot(x + 1, average_price)
        plt.title("Daily average price evolution")
        plt.xlabel("Iteration number")
        plt.ylabel("Average price")
        plt.show()
