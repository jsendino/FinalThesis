#!/usr/bin/env python
"""
Module with all necessary parameters and functions to define the consumer-side.
"""
import numpy as np

from building_block import BuildingBlock
from constants import Constants
from cost import Cost


class Consumers:
    def __init__(self, num_blocks, num_households):
        self.blocks = np.array(([BuildingBlock(num_households[0])] * num_blocks))
        self.num_total_households = 0
        for i in range(num_blocks):
            self.blocks[i] = BuildingBlock(num_households[i])
            self.num_total_households += self.blocks[i].num_households
            self.blocks[i].assign_initial_customers()

        self.total_expenditures = np.zeros(Constants.day_hours.size)

        self.total_demand = np.zeros(Constants.day_hours.size)

    def compute_total_measures(self):
        for i in range(self.blocks.size):
            self.blocks[i].demand.total_demand_per_hour()
            self.total_demand += self.blocks[i].demand.q_per_hour
            self.total_expenditures += np.sum(self.blocks[i].expenditures, 0)

    def get_demand_by_prod(self, iteration, prod, t):
        total_demand = 0
        for j in range(self.blocks.size):
            total_demand += self.blocks[j].get_demand_by_prod(iteration, prod, t)
        return total_demand

    def decide_provider(self, iteration, global_price):
        for t in range(Constants.day_hours.size):
            price = global_price[:, t]
            changing_customers = []
            for i in range(self.blocks.size):
                # Get list of changing customers of block i
                changing_customers_list, new_producer = self.blocks[i].decide_provider(price, t, iteration)
                # Generate list of same number of tuples as number of changing customers
                # with first element being the block, second being one customer and third the new provider
                changing_customers = changing_customers + \
                                     list(zip([i] * len(changing_customers_list),
                                              changing_customers_list,
                                              [new_producer] * len(changing_customers_list)))

            # Shuffle them so to get a random order in which they will change
            np.random.shuffle(changing_customers)

            for change_tuple in changing_customers:
                # Account change
                self.blocks[change_tuple[0]].change_provider(change_tuple[1],
                                                             change_tuple[2],
                                                             t)
                # Recompute price
                possible_providers = self.blocks[change_tuple[0]].possible_providers
                total_demand = np.zeros(possible_providers.size)
                for prod in range(total_demand.size):
                    total_demand[prod] = self.get_demand_by_prod(iteration,
                                                                 possible_providers[prod],
                                                                 [t])
                new_price = Cost.energy_price(total_demand, self.num_total_households)
                # If after change, prices are still different or the cheapest producer is the same, we assume
                # that customers will continue changing provider. Else, stop process.
                new_cheap_prod = np.where(new_price == np.min(new_price))
                if all(new_price == new_price[0]) or \
                                np.where(change_tuple[2] == possible_providers) != \
                                new_cheap_prod:
                    break
        return
