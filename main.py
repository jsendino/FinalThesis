#!/usr/bin/env python
import numpy as np

from Appliances.air_conditioner import AirConditioner
from battery import Battery
from constants import Constants
from cost import Cost
from demand_response import Demand
from Appliances.entertainment import Entertainment
from Appliances.lightning import Lightning
from Appliances.phev import Phev
from Appliances.washer import Washer

__author__ = 'jorge'


def main():
    Demand.initialize_demand()
    Battery.initialize_battery()

    air = AirConditioner()
    washer = Washer()
    phev = Phev()
    light = Lightning()
    ent = Entertainment()

    appliances = np.array((air, phev, washer, light, ent))

    # Compute final price of the energy (will update the demand too)
    price, house_price = Cost.energy_price(appliances)
    for t in range(0, Constants.day_hours.size):
        Demand.total_demand_per_hour(t)
    #     for i in range(0, Constants.num_households):
    #         AirConditioner.update_temperature(i, t)
    #
    #         Constants.utility[i, 0, t] = AirConditioner.utility(i, t)  # Air conditioner utility for costumer i
    #         Constants.utility[i, 3, t] = Lightning.utility(i, t)
    #         Constants.utility[i, 4, t] = Appliances.entertainment.utility(i, t)
    #

    #
    # Constants.utility[:, 1, 0] = Phev.utility()
    # Constants.utility[:, 2, 0] = Washer.utility()

    Demand.plot_demand()
    Cost.plot_price(price)
    Cost.plot_price(house_price)
    Battery.plot_battery(price)


if __name__ == "__main__":
    main()
