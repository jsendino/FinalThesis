#!/usr/bin/env python
import numpy as np

__author__ = 'jorge'


class Constants:
    """
    Class with general constants and variables used along the project.
    """
    num_producers = 2
    num_households = 25
    num_blocks = 4
    num_appliances = 5

    # House type (0 if house is occupied all day)
    household_type = np.zeros(num_households)
    household_type[int(num_households/2):num_households] = 1

    day_hours = (8 + np.arange(0, 24)) % 24

    # Utility matrix
    utility = np.zeros((num_households, num_appliances, day_hours.size))

    # Constants matrix
    a = np.ones((num_households, num_appliances))
    b = np.ones((num_households, num_appliances))
    c = np.ones((num_households, num_appliances))

    # Max number of iterations to compute cost
    max_num_iterations = 1231

    # Final number of iterations performed when algorithm converged
    final_iteration_number = 0
