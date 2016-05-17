#!/usr/bin/env python
"""
Module with all necessary parameters and functions to define the consumer-side.
"""
import numpy as np

from building_block import BuildingBlock
from constants import Constants
from cost import Cost
from itertools import chain

class Consumers:
    def __init__(self, num_blocks, num_households, possible_providers):
        self.blocks = np.array(([BuildingBlock(num_households[0], 0)] * num_blocks))
        self.num_households = num_households
        self.num_total_households = 0

        for i in range(num_blocks):
            # Flatten element i of possible providers (composed by list of tuples with first element being the main
            # provider and second element being a variable length list with secondary providers
            flat_prov = [possible_providers[i][0]] + list(chain(possible_providers[i][1]))
            self.blocks[i] = BuildingBlock(num_households[i], np.array(flat_prov))
            self.num_total_households += self.blocks[i].num_households
            self.blocks[i].assign_initial_customers()

        self.adjacency_matrix = np.zeros((Constants.day_hours.size,
                                          Constants.num_producers+self.num_total_households,
                                          Constants.num_producers+self.num_total_households))

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
        # self.reset_adj_matrix()
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
                try:
                    self.blocks[change_tuple[0]].change_provider(change_tuple[1],
                                                                 change_tuple[2],
                                                                 t)
                except:
                    print(iteration,t)
                    exit(1)
                # Recompute price
                possible_providers = self.blocks[change_tuple[0]].possible_providers
                total_demand = np.zeros(possible_providers.size)
                for prod in range(total_demand.size):
                    total_demand[prod] = self.get_demand_by_prod(iteration,
                                                                 possible_providers[prod],
                                                                 [t])
                new_price = Cost.energy_price(total_demand, self.num_total_households)

                # Take new producer and block and update adjacency matrix
                # self.update_adj_matrix(t, change_tuple[2], change_tuple[0])

                # If after change, prices are still different or the cheapest producer is the same, we assume
                # that customers will continue changing provider. Else, stop process.
                new_cheap_prod = np.where(new_price == np.min(new_price))
                if all(new_price == new_price[0]) or \
                                np.all(np.where(change_tuple[2] == possible_providers)[0] != new_cheap_prod):
                    break
            self.update_adj_matrix(t)
        return

    def get_initial_bids(self, iteration, t):
        bids = np.zeros(self.num_total_households)

        # Columns corresponding to neighbors in block i in global demand matrix
        accumulated_houses = np.insert(np.cumsum(self.num_households), 0, 0)
        accumulated_houses = accumulated_houses[:len(accumulated_houses)-1]

        for i in range(Constants.num_blocks):
            bids[accumulated_houses[i]:
            (accumulated_houses[i] + self.num_households[i])] = self.blocks[i].get_initial_bids(iteration, t)

        return bids

    def update_adj_matrix(self, t):
        # Rows representing nodes to which add neighbors (producers in this case)
        row_range = range(0, Constants.num_producers)
        for i in range(Constants.num_blocks):
            # Subset the part of the adjacency matrix of block i with represents the edges between households
            # and producers
            matrix = self.blocks[i].adjacency_matrix[row_range,
                                                     Constants.num_producers:
                                                     (Constants.num_producers + self.num_households[i]), t]
            # Columns corresponding to neighbors in block i in global adjacency matrix
            accumulated_houses = np.insert(np.cumsum(self.num_households), 0, 0)
            accumulated_houses = accumulated_houses[:len(accumulated_houses)-1]

            self.adjacency_matrix[t,
                                  row_range,
                                  (Constants.num_producers + accumulated_houses[i]):
                                  (Constants.num_producers + accumulated_houses[i] + self.num_households[i])] = matrix

    # def reset_adj_matrix(self):
    #     self.adjacency_matrix = np.zeros((Constants.day_hours.size,
    #                                       Constants.num_producers+Constants.num_blocks,
    #                                       Constants.num_producers+Constants.num_blocks))