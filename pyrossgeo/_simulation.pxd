import numpy as np
cimport numpy as np

from pyrossgeo.__defs__ cimport node, cnode, transporter, DTYPE_t
from pyrossgeo.Simulation cimport Simulation

cdef simulate(Simulation self, DTYPE_t[:] X_state, DTYPE_t t_start, DTYPE_t t_end, object _dts, int steps_per_save=*,
                            str save_path=*, bint only_save_nodes=*, int steps_per_print=*,
                            object event_times=*, object event_function=*)
                            #object cevent_times=*, SIM_EVENT cevent_function=*)