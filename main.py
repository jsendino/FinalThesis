#!/usr/bin/env python
import numpy as np

from constants import Constants
from cost import Cost
from consumers import Consumers

__author__ = 'jorge'


def main():

    np.random.seed(18765)
    consumers = Consumers(Constants.num_blocks, [25] * Constants.num_blocks)

    # Compute final price of the energy (will update the demand too)
    price = Cost.optimal_demand_response(consumers)

    # for i in range(b.size):
    #     b[i].plot_demand()

    # Cost.plot_price(price)
    # #Cost.plot_price(house_price)
    # Battery.plot_battery(Cost.price)

    consumers.compute_total_measures()

    np.savetxt("expenditures.txt", consumers.total_expenditures)
    np.savetxt("demand.txt", consumers.total_demand)

if __name__ == "__main__":
    main()
