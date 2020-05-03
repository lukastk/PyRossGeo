from pyrossgeo.csimulation cimport DTYPE_t, SIM_EVENT, SIM_EVENT_NULL, csimulation, node, cnode, transporter
from pyrossgeo.csimulation import DTYPE

cdef csimulate(csimulation self, DTYPE_t[:] X_state, DTYPE_t t_start, DTYPE_t t_end, object _dts, int steps_per_save=*,
                            str out_file=*, int steps_per_print=*, bint only_save_nodes=*,
                            object event_times=*, object event_function=*,
                            object cevent_times=*, SIM_EVENT cevent_function=*)