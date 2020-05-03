from pyrossgeo.csimulation cimport DTYPE_t, csimulation, node, cnode, transporter
from pyrossgeo.csimulation import DTYPE

import numpy as np
cimport numpy as np

cpdef compute(csimulation self, DTYPE_t[:] X_state, DTYPE_t[:] dX_state, DTYPE_t t, DTYPE_t dt):
    t_start = t
    dts = np.array(dt)
    steps = 1
    steps_per_save = -1
    out_file = ''
    print_per_n_steps = -1
    
    self._simulate(self, X_state, dX_state, t_start, steps, dts, steps_per_save, out_file, print_per_n_steps)

cpdef free_sim(self): # TODO
    """for ni in range(self.nodes_num):
        free(self.nodes[ni].incoming_T_indices)
        free(self.nodes[ni].outgoing_T_indices)
    free(self.nodes)

    free(self.cnodes)

    for ti in range(self.Ts_num):
        free(self.Ts[ti].T)
        #free(self.Ts[ti].schedule)
    free(self.Ts)

    for cti in range(self.cTs_num):
        free(self.cTs[cti].T)
        #free(self.cTs[cti].schedule)
    free(self.cTs)
    """
    pass