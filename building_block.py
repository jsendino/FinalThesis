#!/usr/bin/env python
"""
Module with all necessary parameters and functions to define a block of buildings.
"""
import numpy as np
from battery import Battery
from constants import Constants
from cost import Cost
from demand_response import Demand
from Appliances.air_conditioner import AirConditioner
from Appliances.entertainment import Entertainment
from Appliances.lightning import Lightning
from Appliances.phev import Phev
from Appliances.washer import Washer

__author__ = 'jorge'


class BuildingBlock:
    def __init__(self, num_households):
        self.num_households = num_households
        self.num_appliances = 5

        # House type (0 if house is occupied all day)
        self.household_type = np.zeros(self.num_households)
        self.household_type[int(self.num_households / 2):self.num_households] = 1

        # Constants matrix
        self.a = np.ones((self.num_households, self.num_appliances))
        self.b = np.ones((self.num_households, self.num_appliances))
        self.c = np.ones((self.num_households, self.num_appliances))

        self.battery = Battery(self.num_households)
        self.demand = Demand(self.num_households, self.num_appliances)

        # Arrays with energy prices
        self.price = np.zeros((Constants.max_num_iterations, Constants.num_producers, Constants.day_hours.size))

        # Logical array with houses consuming of each producer
        self.customers = np.zeros((Constants.num_producers, self.num_households, Constants.day_hours.size), 'int')
        self.adjacency_matrix = np.zeros((Constants.num_producers + self.num_households,
                                          Constants.num_producers + self.num_households), 'int')
        self.possible_providers = np.sort(np.random.choice(range(Constants.num_producers),
                                                           size=2,
                                                           replace=False))

        # Money each house spends in energy by hour
        self.expenditures = np.zeros((self.num_households, Constants.day_hours.size))

        air = AirConditioner(self.num_households)
        washer = Washer()
        phev = Phev()
        light = Lightning()
        ent = Entertainment()

        self.appliances = np.array((air, phev, washer, light, ent))

    def get_demand_by_prod(self, iteration, producer, t):
        demand = np.zeros(len(t))
        for i in range(len(t)):
            consumers = np.nonzero(self.customers[producer, :, t[i]])[0]
            demand[i] = np.sum(self.demand.total_house_demand[iteration, consumers, t[i]])
        return demand

    def assign_initial_customers(self):
        # Initialize vector that indicates to which producer a house is buying energy. It will be producer 0.
        vector = np.zeros(Constants.num_producers)
        vector[0] = 1
        # For each house, randomly shuffle that vector so that they have the same probability. Make it constant
        # within all day

        for i in range(self.num_households):
            self.customers[:, i] = np.tile(np.random.permutation(vector), (24, 1)).T
            # Create adjacency matrix
            # cls.array_to_adjacency_mat(cls.customers)

    def decide_provider(self, global_price, t, iteration):
        # Consider price only from those providers that are reachable
        possible_price = global_price[self.possible_providers]
        cheapest_producer = np.where(global_price == np.min(possible_price))[0]
        expensive_producers = np.setdiff1d(self.possible_providers,
                                           cheapest_producer)

        # Find customers that are going to change producer
        changing_customers = np.where(self.customers[expensive_producers, :, t])[1]
        # From those that will benefit from changing, take only those that demand energy in hour t
        consuming = np.where(self.demand.total_house_demand[iteration, :, t] > 0)
        changing_customers = np.intersect1d(changing_customers, consuming)
        return changing_customers, cheapest_producer

    def change_provider(self, changing_customers, new_producer, t):
        old_producer = self.customers[:, changing_customers, t].nonzero()

        self.customers[old_producer, changing_customers, t], self.customers[new_producer, changing_customers, t] = \
            self.customers[new_producer, changing_customers, t], self.customers[old_producer, changing_customers, t]

    def update_parameters(self, iteration, price, appliances):
        for house in range(self.num_households):
            # Get provider of current house
            prod = np.nonzero(self.customers[:, house])
            # Optimize demand and battery
            self.optimize_parameters(iteration, price[prod], house, appliances)
            # Use battery to support demand
            self.use_battery(iteration, house)
            # Update expenditures
            self.expenditures[house] = self.demand.total_house_demand[iteration+1, house] * price[prod]

    def optimize_parameters(self, iteration, price, household, appliances):
        self.demand.adapt_demand(household, int(self.household_type[household]), iteration, price, appliances)
        self.adapt_charge_rate(iteration, price, household)

    def adapt_charge_rate(self, iteration, price, household):
        self.battery.adapt_charge_rate(household, iteration, price)
        self.check_demand_requirements(iteration, household)

    def check_demand_requirements(self, iteration, household):
        """
        Check whether the state of the battery meets requirements regarding total demand.
        If not, modify charge so it does.
        :param iteration: number of iteration
        :param household: index of current household in household matrix
        """
        # Find hours in which battery charge rate makes demand go negative
        null_demand_hours = np.logical_and(self.demand.total_house_demand[iteration + 1, household] <
                                           abs(self.battery.charge_rate[household]),
                                           self.battery.charge_rate[household] < 0)
        self.battery.charge_rate[household, null_demand_hours] = \
            - self.demand.total_house_demand[iteration + 1, household, null_demand_hours]

    def use_battery(self, iteration, household):
        """
        Function that takes into account the use of the battery: if it is charging this charge rate will be added, else
        it will be subtracted
        :param iteration:
        """
        self.demand.total_house_demand[iteration + 1, household] += self.battery.charge_rate[household]
        # If demand is lower that 0 in some cases, it means that battery must be stopped in those hours
        null_demand_hours = self.demand.total_house_demand[iteration + 1, household] < 0
        # Set demand to 0
        self.demand.total_house_demand[iteration + 1, household, null_demand_hours] = 0

    def sell_battery(self, iteration, household, t, charge):
        """
        Sell amount of charge from battery. This method is called when resaling available charge, so some checks
        that are performed when using the battery are not needed.
        :rtype: integer
        :param iteration: number of iteration
        :param household: index of current household in household matrix
        :param t: hour of day
        :param charge: amount of charge to increment
        :return: actual incremented charge according to working requirements
        """
        return self.battery.change_battery_level(iteration, household, t, charge)

    def start_battery_market(self, iteration, market_price):
        """
        Start the process of resaling charge from battery.
        :param iteration: number of iteration
        :param market_price: price the market has set for current iteration
        """
        final_price = np.zeros(Constants.day_hours.size)
        for t in range(0, Constants.day_hours.size):
            # Boolean array indicating which houses have consumed above its battery's capacity
            buying_houses, required_demand, selling_houses, supply = self.get_market_participants(iteration, t)
            if all(supply) == 0:
                final_price[t] = float('nan')
                continue
            # Given the houses that are going to sell energy, compute for each one the new market_price
            # that is going to be selling at
            house_price = np.zeros(selling_houses.size)
            for i in range(0, selling_houses.size):
                # This price is the mean cost of consuming its scheduled energy demand
                house_price[i] = self.compute_resell_price(market_price, i, iteration, t)

            new_house_price, new_selling_houses, supply = self.filter_selling_houses(house_price, market_price,
                                                                                     selling_houses, supply, t)

            if len(supply) == 0:
                continue
            # If demand overcomes supply, all sellers are going to sell all their charge. As they do not have to compete
            # with their neighbours, price can just be set a differential below the market price
            if required_demand >= np.sum(supply):
                final_price[t] = market_price[t] - Cost.delta
                actual_supply = self.sell_battery(iteration, new_selling_houses, t, -supply)
                self.demand.increment_total_demand(iteration + 1, buying_houses, t,
                                                   -np.sum(abs(actual_supply)) / buying_houses.size)

                self.increment_expenditures(buying_houses,
                                            new_selling_houses,
                                            actual_supply,
                                            market_price[t], final_price[t], t)
            # Else, sellers have to compete. Buyers will start buying from the lowest price house until all demand is
            # fulfilled. The last selling house will be the one to set the price. For all the rest houses, it will be
            # enough to set the price one differential below that last seller's price.
            else:
                # Order houses based in lower market_price
                new_selling_houses = new_selling_houses[new_house_price.argsort()]
                supply_ordered = supply[new_house_price.argsort()]
                new_house_price.sort()
                # Find which is the seller that is the last to sell (that seller that fulfills all required demand)
                last_seller = np.where(np.cumsum(supply_ordered) >= required_demand)[0][0]

                final_price[t] = new_house_price[last_seller] - Cost.delta
                if last_seller != 0:
                    # Perform charge exchange
                    most_sellers_supply = self.sell_battery(iteration,
                                                            new_selling_houses[range(0, last_seller)],
                                                            t,
                                                            - supply_ordered[range(0, last_seller)])
                    # Update expenditures for energy bought to sellers with lower price
                    self.increment_expenditures(buying_houses,
                                                new_selling_houses[range(0, last_seller)],
                                                most_sellers_supply,
                                                market_price[t], final_price[t], t)
                last_seller_supply = self.sell_battery(iteration,
                                                       new_selling_houses[last_seller],
                                                       t,
                                                       - (
                                                           required_demand - np.sum(
                                                               supply_ordered[range(0, last_seller)])))
                # As demand is lower than supply, all demand is fulfilled
                self.demand.increment_total_demand(iteration + 1, buying_houses, t, 0)

                last_seller_price = Cost.delta + new_house_price[last_seller]

                self.increment_expenditures(buying_houses,
                                            new_selling_houses[last_seller],
                                            last_seller_supply,
                                            market_price[t], last_seller_price, t)

        return final_price

    def filter_selling_houses(self, house_price, market_price, selling_houses, supply, t):
        """
        Regarding the resaling market, eliminates from selling houses those which may not benefit from selling
        :param house_price: sell price each house has calculated
        :param market_price: price the market has set for current iteration
        :param selling_houses: array with houses that may sell charge
        :param supply: charge supply from those selling houses
        :param t: hour of the day in which market is to take place
        :return: tuple with the new selling houses, their supply and selling price
        """
        # Compare the new prices to that of the market and remove from the process those houses whose prices is
        # always lower than that of the market (will not recover from sell)
        profit_houses = np.where(np.any(house_price.reshape((house_price.size, 1)) >
                                        np.tile(market_price[range(t + 1, Constants.day_hours.size)],
                                                (house_price.size, 1)),
                                        1))[0]
        # Subset also those houses with battery surplus at the end of the day
        surplus_houses = np.sum(self.battery.charge_rate[np.ix_(selling_houses,
                                                                range(0, Constants.day_hours.size))], 1) > 0

        # Houses that contribute to resale will be those that have battery surplus and can recover the battery
        # that are selling that day
        new_selling_houses = np.union1d(selling_houses[profit_houses],
                                        selling_houses[surplus_houses])

        lower_price_houses = np.where(house_price < market_price[t])[0]
        new_selling_houses = np.intersect1d(new_selling_houses,
                                            selling_houses[lower_price_houses])
        new_house_price = house_price[np.where(np.in1d(selling_houses, new_selling_houses))]
        # Get available charge after subset
        supply = supply[np.where(np.in1d(selling_houses, new_selling_houses))]

        return new_house_price, new_selling_houses, supply

    def get_market_participants(self, iteration, t):
        """
        In a given iteration and hours, gets all households that are eligible to participate in a charge resaling
        market either as seller or as buyers.
        :param iteration: number of iteration
        :param t: hour of the day in which market is to take place
        :return: tuple with buying houses, required demand from those houses, selling houses and supply
        """
        discharging_houses = np.where(np.logical_and(self.battery.charge_rate[:, t] < 0,
                                                     self.battery.charge_level[iteration + 1, :, t] <
                                                     abs(self.battery.charge_rate[:, t])))[0]
        # Find demand thar is needed in those houses
        required_demand = np.sum(self.demand.total_house_demand[iteration + 1, discharging_houses, t])
        # Get houses with available charge in their battery
        charging_houses = np.where(self.battery.charge_rate[:, t] > 0)[0]
        charged_houses = np.where(self.battery.charge_level[iteration + 1, :, t] > abs(self.battery.charge_rate[:, t]))[
            0]
        # If no charge is left in any house or not demand is needed, skip
        if charging_houses.size + charged_houses.size == 0 or not required_demand:
            # continue
            supply = np.zeros(self.num_households)
        else:
            # Else, get available charge after subset
            supply = np.zeros((charging_houses.size + charged_houses.size))
            for i, house in enumerate(np.union1d(charging_houses, charged_houses)):
                supply[i] = self.battery.available_selling_charge(iteration, house, t)

        selling_houses = np.union1d(charged_houses, charging_houses)

        return discharging_houses, required_demand, selling_houses, supply

    def compute_resell_price(self, market_price, household, iteration, t):
        try:
            return np.average(market_price[range(0, t + 1)],
                              weights=self.demand.total_house_demand[iteration + 1, household, range(0, t + 1)] +
                                      self.battery.charge_rate[household, range(0, t + 1)])
        except ZeroDivisionError:
            return np.mean(market_price[range(0, t + 1)])

    def increment_expenditures(self, buying_houses, selling_houses, amount, market_price, final_price, t):
        self.expenditures[selling_houses, t] -= abs(amount) * final_price
        self.expenditures[buying_houses, t] -= (np.sum(abs(amount)) / buying_houses.size) * \
                                               (market_price - final_price)

    def plot_demand(self):
        self.demand.plot_demand(self.num_households, self.household_type, self.num_appliances, self.battery.charge_rate)
