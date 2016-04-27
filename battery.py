#!/usr/bin/env python
"""
Module with all necessary parameters to define the battery. It also contains the function to obtain the actual
level of charge of the battery.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from constants import Constants

__author__ = 'jorge'


class Battery:
    def __init__(self, num_households):
        # Maximum charge level (in Wh) and charge rate
        self.storage_cap = 800
        self.max_charging_rate = 250  # Maximum charging/discharging rates (in watts)
        self.charge_rate = np.zeros((num_households, len(Constants.day_hours)))

        # Charge rate with which the battery will be initialized
        self.initial_charge_rate = np.zeros(24)

        # Max charge level
        self.min_charge_level = 0
        self.charge_level = np.zeros((Constants.max_num_iterations, num_households, len(Constants.day_hours)))
        self.initial_charge_level = 100
        self.total_charge_level = np.zeros((Constants.max_num_iterations, num_households))

        # Battery constants
        self.H1 = 5 * np.exp(-7)
        self.H2 = 4 * np.exp(-7)
        self.H3 = 1
        self.delta = 0.2

        # gamma is the constant representing the degree of responsiveness of the costumer to a change in demand price
        self.gamma = 0.1
        self.c = np.ones(num_households)

        self.initialize_battery(num_households)

    def initialize_battery(self, num_households):
        """
        Initialize the battery schedule of all household to a default value that will be adapted
        """

        self.charge_rate[range(0, num_households)] = self.initial_charge_rate
        self.charge_rate[range(0, num_households)] = np.zeros(24)
        self.charge_level[0] = self.initial_charge_level + np.cumsum(self.charge_rate, 1)

    def compute_operating_cost(self, house, iteration):
        """
        Computes the cost of operating the battery the whole day in a determined iteration for a given house
        :param iteration: number of iteration in which the cost is computed
        :param house: integer used as index of the costumer who the cost is retrieved of
        :rtype : integer
        :return: returns the cost of operating the battery belonging to one costumer
        """
        cost = self.operating_cost(self.charge_rate[house],
                                   self.H1, self.H2, self.H3, self.delta, self.storage_cap, self.c[house],
                                   self.charge_level[iteration, house])
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

    def adapt_charge_rate(self, household, iteration, price):
        """
        Function that adapts the charge rate schedule of the battery for one household.
        It takes the energy price broadcasted by the producer in current iteration and runs an
        optimization function

        :param household: house that is running the adaptation process
        :param iteration: iteration of the adaptation algorithm
        :param price: array with energy price in every hour of the day
        """
        operating_cost_derivative = scipy.misc.derivative(self.operating_cost,
                                                          self.charge_rate[household],
                                                          dx=1,
                                                          args=(self.H1, self.H2, self.H3, self.delta,
                                                                self.storage_cap, self.c[household],
                                                                self.charge_level[iteration, household],))

        # Compute new adapted charging schedule
        self.change_battery_level(iteration,
                                  household,
                                  range(0, Constants.day_hours.size),
                                  -self.gamma * (operating_cost_derivative + price))

    def change_battery_level(self, iteration, household, t, charge):
        """
        Modifies the charge level of the battery, performing the necessary checks
        :rtype: integer
        :param iteration: number of iteration
        :param household: index of current household in household matrix
        :param t: hour of day
        :param charge: amount of charge to increment
        :return: actual incremented charge according to working requirements
        """
        initial_battery_rate = self.charge_rate[household, t]
        # Increment charge
        self.charge_rate[household, t] += charge
        # Modify increment so that it meets requirements
        self.check_bat_requirements(iteration, household)
        # Return actual increment
        return self.charge_rate[household, t] - initial_battery_rate

    def check_bat_requirements(self, iteration, household):
        """
        Check whether the state of the battery meets requirements. If not, modify charge so it does.
        :param iteration: number of iteration
        :param household: index of current household in household matrix
        """
        try:
            # If any of the new charging rate exceeds maximum, truncate it
            exceeded_hours = np.where(abs(self.charge_rate[household]) > self.max_charging_rate)
            self.charge_rate[household, exceeded_hours] = np.sign(self.charge_rate[household, exceeded_hours]) * \
                                                          self.max_charging_rate

            # Now that charging rates have been computed, calculate charging level from it
            self.charge_level[iteration + 1, household] = self.initial_charge_level + \
                                                          np.cumsum(self.charge_rate[household])

            # Find hours in which the level has exceed the maximum
            exceeded_hours = np.where(self.charge_level[iteration + 1, household] > self.storage_cap)[0]
            # In those hours, reduce level to maximum
            self.charge_level[iteration + 1, household, exceeded_hours] = self.storage_cap
            # Set rate in those hours to the value needed to reach the max charge
            self.charge_rate[household, exceeded_hours] = self.charge_level[iteration + 1, household, exceeded_hours] - \
                                                          self.charge_level[iteration + 1, household,
                                                                            exceeded_hours - 1]
            # Same for minimum. There is no need here to re-set the rate cause if the level is under 0,
            # it means that the energy is taken out of the battery right when it enters
            # (this fact is taken into account when computing demand)
            insufficient_hours = np.where(self.charge_level[iteration + 1, household] < self.min_charge_level)[0]
            self.charge_level[iteration + 1, household, insufficient_hours] = self.min_charge_level
        except:
            exceeded = np.where(abs(self.charge_rate[household]) > self.max_charging_rate)
            self.charge_rate[household[exceeded[0]], exceeded[1]] = np.sign(self.charge_rate[household[exceeded[0]],
                                                                                             exceeded[1]]) * \
                                                                    self.max_charging_rate
            # Find hours in which the level has exceed the maximum
            exceeded = np.where(self.charge_level[iteration + 1, household] > self.storage_cap)
            # In those hours, reduce level to maximum
            self.charge_level[iteration + 1, exceeded[0], exceeded[1]] = self.storage_cap
            # Set rate in those hours to the value needed to reach the max charge
            self.charge_rate[household[exceeded[0]], exceeded[1]] = self.charge_level[iteration + 1,
                                                                                      household[exceeded[0]],
                                                                                      exceeded[1]] - \
                                                                    self.charge_level[
                                                                        iteration + 1, household[exceeded[0]],
                                                                        exceeded[1] - 1]
            # Same for minimum. There is no need here to re-set the rate cause if the level is under 0,
            # it means that the energy is taken out of the battery right when it enters
            # (this fact is taken into account when computing demand)
            insufficient = np.where(self.charge_level[iteration + 1, household] < self.min_charge_level)
            self.charge_level[iteration + 1, household[insufficient[0]], insufficient[1]] = self.min_charge_level

    def available_selling_charge(self, iteration, household, t):
        charge = 0

        if self.charge_rate[household, t] > 0:
            charge += self.charge_rate[household, t]

        if self.charge_level[iteration + 1, household, t] > abs(self.charge_rate[household, t]):
            charge += self.charge_level[iteration + 1, household, t]

        if charge > self.max_charging_rate:
            charge = self.max_charging_rate + self.charge_rate[household, t]

        return charge

    def plot_battery(self, price):
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
        plt.step(x, self.initial_charge_rate, color='r', where='post')
        plt.subplot(312, sharex=ax)
        plt.title("Final battery charge rate")
        plt.step(x, self.charge_rate[0], color='b', where='post')
        plt.subplot(313, sharex=ax)
        plt.title("Final price")
        plt.step(x, np.mean(price[Constants.final_iteration_number - 1], 0), color='g', where='post')
        plt.xlim(min(x), max(x) + 1)

        fig.subplots_adjust(hspace=.5)

        plt.show()
