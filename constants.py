#!/usr/bin/env python
import numpy as np

__author__ = 'jorge'


class Constants:
    """
    Class with general constants and variables used along the project.
    """
    num_producers = 2
    num_blocks = 4

    # Constants matrix
    a = np.ones((25, 5))
    b = np.ones((25, 5))
    c = np.ones((25, 5))

    day_hours = (8 + np.arange(0, 24)) % 24

    # Max number of iterations to compute cost
    max_num_iterations = 1231

    # Final number of iterations performed when algorithm converged
    final_iteration_number = 0
