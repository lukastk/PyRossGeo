from pyrossgeo.csimulation cimport DTYPE_t, SIM_EVENT, SIM_EVENT_NULL, csimulation, node, cnode, transporter
from geodemic.csimulation import DTYPE

cdef class simulation:
    cdef public csimulation cg

    cdef csimulate(self, DTYPE_t[:] X_state, DTYPE_t t_start, DTYPE_t t_end, object _dts, int steps_per_save=*,
                                str out_file=*, int steps_per_print=*, bint only_save_nodes=*,
                                int steps_per_event=*,object event_function=*,
                                int steps_per_cevent=*, SIM_EVENT cevent_function=*)