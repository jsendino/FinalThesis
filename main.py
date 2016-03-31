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
    Cost.energy_price(appliances)
    Demand.total_demand_per_hour()

    Demand.plot_demand()
    Cost.plot_price(Cost.price)
    #Cost.plot_price(house_price)
    Battery.plot_battery(Cost.price)

    np.savetxt("expenditures.txt", np.sum(Cost.expenditures, 0))
    np.savetxt("demand.txt", Demand.q_per_hour)
    np.savetxt("cost.txt", Cost.price[Constants.final_iteration_number - 1])


if __name__ == "__main__":
    main()
