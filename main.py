#!/usr/bin/env python
import numpy as np

from Appliances.air_conditioner import AirConditioner
from Appliances.entertainment import Entertainment
from Appliances.lightning import Lightning
from Appliances.phev import Phev
from Appliances.washer import Washer
from battery import Battery
from cost import Cost
from demand_response import Demand

__author__ = 'jorge'


def main():
    #Demand.plot_total_demand()
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
    Demand.total_demand_per_hour()

    Demand.plot_demand()
    Cost.plot_price(price)
    Battery.plot_battery(price)



if __name__ == "__main__":
    main()
