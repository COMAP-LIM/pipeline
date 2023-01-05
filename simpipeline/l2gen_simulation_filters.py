import warnings
import numpy as np

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)

from l2gen_filters import Filter
from simpipeline.l2_simulations import SimCube


class Signal_Injector(Filter):
    """General signal injection class.

    Args:
        Filter (Filter): Filter super class on which to build signal injector.
    """

    name = "sig2tod"
    name_long = "Signal injector filter"

    def __init__(self, params, simulation: SimCube, *args):
        self.boost = params.boost_factor
        self.simulation = simulation
