#!/usr/bin/env python
"""
Module with all necessary parameters to define the battery. It also contains the function to obtain the actual
level of charge of the battery.
"""
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

from constants import Constants

__author__ = 'jorge'


class Battery:
    # Maximum charge level (in Wh) and charge rate
    # storage_cap = np.random.uniform(5500, 6500)  # Expressed in watts/hours
    storage_cap = 800
    max_charging_rate = 250  # Maximum charging/discharging rates (in watts)
    charge_rate = np.zeros((Constants.num_households, len(Constants.day_hours)))
    # Charge rate with which the battery will be initialized
    # charge_rate[range(0, Constants.num_households)] = np.array((1900, 1600, 800, 0, -200, -900, 25, 750,
    #                                                             800, 750, -2500, -2500, -1800, -1500, -1000, -500,
    #                                                             100, 250, 15, 100, 300, 150, 250, 200))
    initial_charge_rate = np.array((210, 190, 100, 0, -20, -100, 0, 150,
                                    175, 200, -200, -200, -150, -125, -90,
                                    -50, 10, 15, 15, 10, 30, 15, 25, 20))
    initial_charge_rate = np.zeros(24)

    # Max charge level
    min_charge_level = 0
    charge_level = np.zeros((Constants.max_num_iterations, Constants.num_households, len(Constants.day_hours)))
    initial_charge_level = 0
    total_charge_level = np.zeros((Constants.max_num_iterations, Constants.num_households))

    # Battery constants
    # H1 = 1
    # H2 = 0.75
    # H3 = 0.5
    H1 = 5 * np.exp(-7)
    H2 = 4 * np.exp(-7)
    H3 = 1
    delta = 0.2

    # gamma is the constant representing the degree of responsiveness of the costumer to a change in demand price
    gamma = 0.1
    c = np.ones(Constants.num_households)

    @classmethod
    def initialize_battery(cls):
        """
        Initialize the battery schedule of all household to a default value that will be adapted
        """

        cls.charge_rate[range(0, Constants.num_households)] = cls.initial_charge_rate
        cls.charge_rate[range(0, Constants.num_households)] = np.zeros(24)
        cls.charge_level[0] = cls.initial_charge_level + np.cumsum(cls.charge_rate, 1)

    @classmethod
    def compute_operating_cost(cls, house, iteration):
        """
        Computes the cost of operating the battery the whole day in a determined iteration for a given house
        :param iteration: number of iteration in which the cost is computed
        :param house: integer used as index of the costumer who the cost is retrieved of
        :rtype : integer
        :return: returns the cost of operating the battery belonging to one costumer
        """
        cost = cls.operating_cost(cls.charge_rate[house],
                                  cls.H1, cls.H2, cls.H3, cls.delta, cls.storage_cap, cls.c[house],
                                  cls.charge_level[iteration, house])
        return cost

    @staticmethod
    def operating_cost(rate, h1, h2, h3, delta, cap, c, charge_level):
        """
        Function that implements the operating cost function of the battery.
        It is implemented outside compute_operating_cost() to provide modularity and to make the calculus of its
        derivative easier
        :param rate: array with charging/discharging schedule of the battery for one household
        :param h1: constant that weights the damaging effect of fast charging/discharging
        :param h2: constant that weights the penalization of charging/discharging cycles
        :param h3: constant that weights the damage of deep discharging
        :param delta: constant that weights the storage capacity of the battery
        :param cap: total storage capacity of the battery (in watts)
        :param c: constant
        :param charge_level: array with charge levels during the day
        :return: cost of operating the battery
        """
        return h1 * np.sum(rate) ** 2 - \
               h2 * np.sum(rate[range(0, len(Constants.day_hours) - 1)] *
                           rate[range(1, len(Constants.day_hours))]) + \
               h3 * np.sum(np.minimum(charge_level - delta * cap, 0)) ** 2 + c

    @classmethod
    def adapt_charge_rate(cls, household, iteration, price):
        """
        Function that adapts the charge rate schedule of the battery for one household.
        It takes the energy price broadcasted by the producer in current iteration and runs an
        optimization function

        :param household: house that is running the adaptation process
        :param iteration: iteration of the adaptation algorithm
        :param price: array with energy price in every hour of the day
        """
        operating_cost_derivative = scipy.misc.derivative(cls.operating_cost,
                                                          cls.charge_rate[household],
                                                          dx=1,
                                                          args=(cls.H1, cls.H2, cls.H3, cls.delta,
                                                                cls.storage_cap, cls.c[household],
                                                                cls.charge_level[iteration, household],))

        # Compute new adapted charging schedule
        cls.charge_rate[household] -= cls.gamma * (operating_cost_derivative + price)
        # Check if the status of the battery is consistent
        cls.check_battery(household, iteration)

    @classmethod
    def check_battery(cls, household, iteration):
        """
        Check both the charge schedule and the charge level of the battery to see if they are consistent. If not,
        adjust them.
        :param household: Household to which check the battery status
        :param iteration: Iteration in which the check is done
        """
        # If any of the new charging rate exceeds maximum, truncate it
        exceeded_hours = np.where(abs(cls.charge_rate[household]) > cls.max_charging_rate)
        cls.charge_rate[household, exceeded_hours] = np.sign(cls.charge_rate[household, exceeded_hours]) * \
                                                     cls.max_charging_rate

        # Now that charging rates have been computed, calculate charging level from it
        cls.charge_level[iteration + 1, household] = cls.initial_charge_level + \
                                                     np.cumsum(cls.charge_rate[household])

        # Find hours in which the level has exceed the maximum
        exceeded_hours = np.where(cls.charge_level[iteration + 1, household] > cls.storage_cap)
        # In those hours, reduce level to maximum
        cls.charge_level[iteration + 1, household, exceeded_hours] = cls.storage_cap
        # Set rate in those hours to the value needed to reach the max charge
        cls.charge_rate[household, exceeded_hours] = cls.charge_level[iteration + 1, household, exceeded_hours] - \
                                                     cls.charge_level[iteration + 1, household,
                                                                      np.subtract(exceeded_hours, 1)]

        # Same for minimum
        insufficient_hours = np.where(cls.charge_level[iteration + 1, household] < cls.min_charge_level)
        cls.charge_level[iteration + 1, household, insufficient_hours] = cls.min_charge_level
        # There is no need here to re-set the rate cause if the level is under 0, it means that the energy is taken out
        # of the battery right when it enters (this fact is taken into account when computing demand)

    @classmethod
    def share_battery(cls, iteration, market_price):
        """

        :param iteration:
        :param market_price:
        """
        from demand_response import Demand
        from cost import Cost
        final_price = np.zeros(Constants.day_hours.size)
        for t in range(0, Constants.day_hours.size):
            # Boolean array indicating which houses have consumed above its battery's capacity
            discharging_houses, required_demand, selling_houses, supply = cls.get_market_participants(iteration, t)

            if all(supply) == 0:
                final_price[t] = float('nan')
                continue

            # Given the houses that are going to sell energy, compute for each one the new market_price
            # that is going to be selling at
            house_price = np.zeros(selling_houses.size)
            for i in range(0, selling_houses.size):
                house_price[i] = np.average(Cost.compute_price(np.sum(Demand.total_house_demand[i], 0)))

            new_house_price, new_selling_houses, supply = cls.filter_selling_houses(house_price, market_price,
                                                                                    selling_houses, supply, t)

            # If demand overcomes supply, all sellers are going to sell all their charge. As they do not have to compete
            # with their neighbours, price can just be set a differential below the market price
            if required_demand >= np.sum(supply):
                final_price[t] = market_price - Cost.delta
                cls.charge_rate[new_selling_houses, t] -= supply
                cls.charge_rate[discharging_houses, t] += np.sum(supply) / discharging_houses.size
            # Else, sellers have to compete. Buyers will start buying from the lowest price house until all demand is
            # fulfilled. The last selling house will be the one to set the price. For all the rest houses, it will be
            # enough to set the price one differential below that last seller's price.
            else:
                # Order houses based in lower market_price
                selling_houses = new_selling_houses[new_house_price.argsort()]
                supply_ordered = supply[new_house_price.argsort()]
                new_house_price.sort()
                # Find which is the seller that is the last to sell (that seller that fulfills all required demand)
                last_seller = np.where(np.cumsum(supply_ordered) >= required_demand)[0][0]

                final_price[t] = new_house_price[last_seller] - Cost.delta

                # Perform charge exchange
                cls.charge_rate[selling_houses[range(0, last_seller)], t] -= supply_ordered[range(0, last_seller)]
                cls.charge_rate[selling_houses[last_seller], t] -= required_demand - np.sum(
                    supply_ordered[range(0, last_seller)])

                cls.charge_rate[discharging_houses, t] = 0

            # for seller in selling_houses:
            #     # Add that charge to the buyer house
            #     for buyer in discharging_houses:
            #         if cls.charge_rate[seller, t] > abs(cls.charge_rate[buyer, t]):
            #             sold_charge = cls.charge_rate[buyer, t]
            #         else:
            #             sold_charge = Battery.charge_rate[seller, t]
            #
            #         cls.charge_rate[buyer, t] += sold_charge
            #         # If current buyer has enough charge, remove it from vector of buyers
            #         if cls.charge_rate[buyer, t] >= 0:
            #             np.delete(discharging_houses, buyer)
            #         supply[seller == selling_houses] -= sold_charge
            #         if supply[seller] == 0:
            #             break

            # # Take charge from those houses that are charging the batt
            # cls.charge_rate[charging_houses, t] -= required_demand / np.sum(charging_houses)
            # # Stop at zero
            # exceeded = np.where(cls.charge_rate[charging_houses, t] < 0)
            # if np.any(exceeded):
            #     # If there are hours in which 0 batt is reached in houses that are sharing charge,
            #     # redistribute available charge
            #     given_charge = required_demand + np.sum(cls.charge_rate[exceeded, t])
            #     cls.charge_rate[charging_houses, t] = 0
            #     cls.charge_rate[discharging_houses, t] = given_charge / np.sum(discharging_houses)
            # else:
            #     # Else, split charge equally
            #     cls.charge_rate[discharging_houses, t] = required_demand / np.sum(discharging_houses)

        for household in range(0, Constants.num_households):
            Battery.check_battery(household, iteration)
            Demand.use_battery(household, iteration)

        return final_price

    @classmethod
    def filter_selling_houses(cls, house_price, market_price, selling_houses, supply, t):
        # Compare the new prices to that of the market and remove from the process those houses whose prices is
        # always lower than that of the market (will not recover from sell)
        lower_price_houses = np.where(np.any(house_price.reshape((house_price.size, 1)) >
                                             np.tile(market_price[range(t + 1, Constants.day_hours.size)],
                                                     (house_price.size, 1)),
                                             1))[0]
        # Subset also those houses with battery surplus at the end of the day
        surplus_houses = np.sum(cls.charge_rate[np.ix_(selling_houses,
                                                       range(t, Constants.day_hours.size))], 1) > 0
        # Houses that contribute to re-sell will be those that have battery surplus and can recover the battery
        # that are selling that day
        new_selling_houses = np.union1d(selling_houses[lower_price_houses],
                                        selling_houses[surplus_houses])
        new_house_price = house_price[np.where(np.in1d(new_selling_houses, selling_houses))]
        # Get available charge after subset
        supply = supply[np.where(np.in1d(new_selling_houses, selling_houses))]

        return new_house_price, new_selling_houses, supply

    @classmethod
    def get_market_participants(cls, iteration, t):
        discharging_houses = np.where(np.logical_and(cls.charge_rate[:, t] < 0,
                                                     cls.charge_level[iteration + 1, :, t] <
                                                     abs(cls.charge_rate[:, t])))[0]
        # Find demand thar is needed in those houses
        required_demand = abs(np.sum(cls.charge_rate[discharging_houses, t] -
                                     cls.charge_level[iteration + 1, discharging_houses, t]))
        # Get houses with available charge in their battery
        charging_houses = np.where(cls.charge_rate[:, t] > 0)[0]
        charged_houses = np.where(cls.charge_level[iteration + 1, :, t] > abs(cls.charge_rate[:, t]))[0]
        # If no charge is left in any house or not demand is needed, skip
        if charging_houses.size + charged_houses.size == 0 or not required_demand:
            # continue
            supply = np.zeros(Constants.num_households)
        else:
            # Else, get available charge after subset
            supply = np.zeros((charging_houses.size + charged_houses.size))
            for i, house in enumerate(np.union1d(charging_houses, charged_houses)):
                # If is charging the battery, all new charge will be available to sell
                if np.any(house == charging_houses):
                    supply[i] = cls.charge_rate[house, t]
                # Else, only battery that is not used (is not discharged) will be available
                elif np.any(house == charged_houses):
                    supply[i] = cls.charge_level[iteration + 1, house, t] - abs(cls.charge_rate[house, t])
        selling_houses = np.union1d(charged_houses, charging_houses)
        return discharging_houses, required_demand, selling_houses, supply

    @classmethod
    def plot_battery(cls, price):
        """
        Plots three figures: one with the initial schedule of the battery (before adjustment), final schedule
        and final price (after adjustment both)
        :param price: Bi-dimensional price matrix with iterations as rows and hours in day as columns
        """
        # Set x axis values
        ax = plt.gca()
        x = range(0, 24)
        ax.set_xticks(x)
        # Set x axis labels
        labels = [str(hour) for hour in Constants.day_hours]
        ax.set_xticklabels(labels)
        # Plot initial charging rate, final charging rate and price
        fig = plt.figure(1)
        plt.subplot(311, sharex=ax)
        plt.title("Initial battery charge rate")
        plt.step(x, cls.initial_charge_rate, color='r', where='post')
        plt.subplot(312, sharex=ax)
        plt.title("Final battery charge rate")
        plt.step(x, Battery.charge_rate[0], color='b', where='post')
        plt.subplot(313, sharex=ax)
        plt.title("Final price")
        plt.step(x, price[Constants.final_iteration_number - 1], color='g', where='post')
        plt.xlim(min(x), max(x) + 1)

        fig.subplots_adjust(hspace=.5)

        plt.show()
