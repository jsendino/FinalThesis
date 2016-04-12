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
    # Arrays with energy prices
    price = np.zeros((Constants.max_num_iterations, Constants.num_producers, Constants.day_hours.size))
    # Array with price at which the energy is re-selled
    house_price = np.zeros((Constants.max_num_iterations, Constants.day_hours.size))

    # Logical array with houses consuming of each producer
    customers = np.zeros((Constants.num_producers, Constants.num_households, Constants.day_hours.size), 'int')
    adjacency_matrix = np.zeros((Constants.num_producers + Constants.num_households,
                                 Constants.num_producers + Constants.num_households), 'int')

    # Modifier accounting for the magnitude of a real population to multiply total energy demand
    magnitude = 5 * 10 ** 4
    #magnitude = 1 * 10 ** 6

    # Modifier that indicates the units in every step of piecewise function
    M = 1 * 10 ** 6

    # Money each house spends in energy by hour
    expenditures = np.zeros((Constants.num_households, Constants.day_hours.size))

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
    def energy_price(cls, total_demand):

        """
        Function that computes the energy price as the marginal cost of energy (the derivative of the
        cost function calculated over the aggregated demand)

        :param total_demand: integer with the aggregated demand of all houses in one hour
        :return: integer with the energy price corresponding to total_demand
        """
        return scipy.misc.derivative(cls.total_cost, total_demand, dx=0.1)

    @classmethod
    def compute_price(cls, iteration, prod):
        price = np.zeros(Constants.day_hours.size)
        for t in range(0, Constants.day_hours.size):
            # Compute aggregated demand for each provider at hour t
            total_demand = np.sum(Demand.total_house_demand[iteration,
                                                            np.nonzero(cls.customers[prod, :, t])[0],
                                                            t]) * Constants.num_blocks * cls.magnitude
            # Price is set to the marginal cost
            price[t] = cls.energy_price(total_demand) / (cls.magnitude * Constants.num_blocks * Constants.num_households)

        return price

    @classmethod
    def compute_resell_price(cls, market_price, household,  iteration, t):
        return np.average(market_price[range(0, t + 1)],
                          weights=Demand.total_house_demand[iteration + 1, household, range(0, t + 1)] +
                                  Battery.charge_rate[household, range(0, t + 1)])

    @classmethod
    def optimal_demand_response(cls, appliances):
        """
        Function that computes the price of a kw of energy during all day.

        It runs an iterative algorithm in which every house adapt its initial demand
        to the cost until equilibrium is reached.

        :param appliances: vector with appliance used in each house
        :rtype : Bi-dimensional array with same rows as iterations ran and same columns as hours in day.
        :return : Each cell of the returned array is the price in that combination of iteration and hour
        """
        i = 0
        while i < Constants.max_num_iterations - 1:
            if i == 0:
                # If initial iteration, assign provider randomly
                cls.assign_initial_customers()
            else:
                # Else, decide provider for every house and hour
                cls.decide_provider(i)

            for prod in range(Constants.num_producers):
                cls.price[i, prod] = cls.compute_price(i, prod)

            # Detect if algorithm has converged. If so, stop iterations
            n_samples_mean = cls.price[range(i - Cost.range, i+1), :]
            if i > cls.range and \
                    np.logical_and(cls.price[i] * (1 - Cost.epsilon) <= n_samples_mean,
                                   n_samples_mean <= cls.price[i] * (1 + Cost.epsilon)).all():
                break
            # For every house, take price for all the day based on the provider decision every house has made
            # and adapt demand
            for j in range(0, Constants.num_households):
                prod = np.nonzero(cls.customers[:, j])[0]
                price = np.zeros(Constants.day_hours.size)
                for t in range(Constants.day_hours.size):
                    price[t] = cls.price[i, prod[t], t]
                Demand.adapt_demand(j, i, price, appliances)
                Battery.adapt_charge_rate(j, i, price)
                Demand.use_battery(j, i)

                cls.expenditures[j] = Demand.total_house_demand[i+1, j] * price
            cls.house_price[i] = Battery.start_battery_market(i, cls.price[i, 0])
            i += 1

        Constants.final_iteration_number = i + 1  # i + 1 cause counting iteration 0

    @classmethod
    def increment_expenditures(cls, buying_houses, selling_houses, amount, market_price, final_price, t):
        cls.expenditures[selling_houses, t] -= abs(amount) * final_price
        cls.expenditures[buying_houses, t] -= (np.sum(abs(amount)) / buying_houses.size) * \
                                              (market_price - final_price)

    @classmethod
    def assign_initial_customers(cls):
        # Initialize vector that indicates to which producer a house is buying energy. It will be producer 0.
        vector = np.zeros(Constants.num_producers)
        vector[0] = 1
        # For each house, randomly shuffle that vector so that they have the same probability. Make it constant
        # within all day
        for i in range(Constants.num_households):
            cls.customers[:, i] = np.tile(np.random.permutation(vector), (24, 1)).T
        # Create adjacency matrix
        #cls.array_to_adjacency_mat(cls.customers)

    @classmethod
    def array_to_adjacency_mat(cls, array):
        cls.adjacency_matrix[0:Constants.num_producers,
                             range(array.shape[0], array.shape[0] + array.shape[1])] = array

    @classmethod
    def decide_provider(cls, iteration):
        # Decision process is been carried in an hour slot
        for t in range(Constants.day_hours.size):
            new_price = cls.price[iteration-1, :, t]

            cheapest_producer = np.where(new_price == np.min(new_price))[0]
            expensive_producers = np.setdiff1d(np.arange(Constants.num_producers),
                                               cheapest_producer)

            # Find customers that are going to change producer
            changing_customers = np.where(cls.customers[expensive_producers, :, t])[1]
            # Sort them so to get an order in which they will change
            np.random.shuffle(changing_customers)

            for i in changing_customers:
                # Account change
                cls.customers[expensive_producers, i, t] = 0
                cls.customers[cheapest_producer, i, t] = 1
                # Recompute price
                new_mean_price = np.zeros(new_price.shape)
                for prod in cheapest_producer.tolist() + expensive_producers.tolist():
                    new_mean_price[prod] = cls.compute_price(iteration, prod)[t]
                # If after change the cheapest producer is the same, we asume that customers will continue changing provider
                # Else, stop process
                if np.where(new_mean_price == np.min(new_mean_price))[0] != cheapest_producer:
                    break

            return

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
        ax.set_yticks(np.arange(min(average_price), max(average_price)+1, max(average_price)/10))

        plt.plot(x+1, average_price)
        plt.title("Daily average price evolution")
        plt.xlabel("Iteration number")
        plt.ylabel("Average price")
        plt.show()
