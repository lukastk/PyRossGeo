from pyrossgeo.csimulation cimport DTYPE_t, SIM_EVENT, SIM_EVENT_NULL, csimulation, node, cnode, transporter
from pyrossgeo.csimulation import DTYPE

from pyrossgeo._initialization import initialize
from pyrossgeo._simulation import simulate
from pyrossgeo._simulation cimport csimulate
from pyrossgeo._helpers import compute

import numpy as np
cimport numpy as np

cdef class simulation:
    def __cinit__(self):
        self.cg = csimulation()

    def get_state_mappings(self):
        return self.cg.state_mappings

    #### Cython methods ####

    def initialize(self, *args, **kwargs):
        return initialize(self.cg, *args, **kwargs)

    def simulate(self, *args, **kwargs):
        return simulate(self.cg, *args, **kwargs)

    cdef csimulate(self, DTYPE_t[:] X_state, DTYPE_t t_start, DTYPE_t t_end, object _dts, int steps_per_save=-1,
                            str out_file="", int steps_per_print=-1, bint only_save_nodes=False,
                            int steps_per_event=-1, object event_function=None,
                            int steps_per_cevent=-1, SIM_EVENT cevent_function=SIM_EVENT_NULL):
        return csimulate(self.cg, X_state, t_start, t_end, _dts, steps_per_save,
                            out_file, steps_per_print, only_save_nodes,
                            steps_per_event, event_function,
                            steps_per_cevent, cevent_function)

    def compute(self, *args, **kwargs):
        compute(self.cg, *args, **kwargs)

    #def get_contact_matrix_keys(self):
    #    return self.cg.get_contact_matrix_keys()

    #def get_contact_matrix(self, str cmat_key):
    #    self.cg.get_contact_matrix(cmat_key)

    #def set_contact_matrix(self, str cmat_key, np.ndarray cmat):
    #    return self.cg.set_contact_matrix(cmat_key, cmat)