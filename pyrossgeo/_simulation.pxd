# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as np

from pyrossgeo.__defs__ cimport node, cnode, transporter, model_term, DTYPE_t
from pyrossgeo.__defs__ import DTYPE
from pyrossgeo.Simulation cimport Simulation

cdef simulate(Simulation self, DTYPE_t[:] X_state, DTYPE_t t_start, DTYPE_t t_end, object _dts, int steps_per_save=*,
                            str save_path=*, bint only_save_nodes=*, int steps_per_print=*,
                            object event_times=*, object event_function=*,
                            int random_seed=*)
                            #object cevent_times=*, SIM_EVENT cevent_function=*)


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed

    cdef cppclass poisson_distribution[T]:
        poisson_distribution()
        poisson_distribution(DTYPE_t a)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"