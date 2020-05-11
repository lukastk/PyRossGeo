from libc.stdlib cimport free

from pyrossgeo.__defs__ cimport node, cnode, transporter, model_term, DTYPE_t
from pyrossgeo.__defs__ import DTYPE
from pyrossgeo.Simulation cimport Simulation

import numpy as np
cimport numpy as np

cpdef compute(Simulation self, DTYPE_t[:] X_state, DTYPE_t[:] dX_state, DTYPE_t t, DTYPE_t dt):
    t_start = t
    dts = np.array(dt)
    steps = 1
    steps_per_save = -1
    out_file = ''
    print_per_n_steps = -1
    
    self._simulate(self, X_state, dX_state, t_start, steps, dts, steps_per_save, out_file, print_per_n_steps)

cpdef free_sim(Simulation self):
    for ni in range(self.nodes_num):
        free(self.nodes[ni].incoming_T_indices)
        free(self.nodes[ni].outgoing_T_indices)
        free(self.nodes[ni].contact_matrix_indices)
        free(self.nodes[ni].linear_coeffs)
        free(self.nodes[ni].infection_coeffs)
    free(self.nodes)

    for ni in range(self.cnodes_num):
        free(self.cnodes[ni].contact_matrix_indices)
        free(self.cnodes[ni].linear_coeffs)
        free(self.cnodes[ni].infection_coeffs)
    free(self.cnodes)

    for ti in range(self.Ts_num):
        free(self.Ts[ti].moving_classes)
    free(self.Ts)

    for cti in range(self.cTs_num):
        free(self.cTs[cti].moving_classes)
    free(self.cTs)

    for age in range(self.age_groups):
        free(self.nodes_at_j_len[age])
        for loc in range(self.max_node_index+1):
            free(self.nodes_at_j[age][loc])
        free(self.nodes_at_j[age])
    free(self.nodes_at_j)
    free(self.nodes_at_j_len)

    for age in range(self.age_groups):
        free(self.cnodes_into_k_len[age])
        for to_k in range(self.max_node_index+1):
            free(self.cnodes_into_k[age][to_k])
        free(self.cnodes_into_k[age])
    free(self.cnodes_into_k)
    free(self.cnodes_into_k_len)

    free(self.model_linear_terms)
    free(self.model_infection_terms)
    