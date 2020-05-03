import numpy as np
cimport numpy as np

DTYPE = np.float # np.float32

cdef class csimulation:
    def __init__(self):
        self.storage = {}

    cpdef get_contact_matrix_keys(self):
        return list(self.contact_matrices_key_to_index.keys())

    cpdef get_contact_matrix(self, str cmat_key):
        return self.contact_matrices[self.contact_matrices_key_to_index[cmat_key]]

    cpdef set_contact_matrix(self, str cmat_key, np.ndarray cmat):
        self.contact_matrices[self.contact_matrices_key_to_index[cmat_key]] = cmat

    cpdef stop_commuting(self, bint s):
        if not self.is_commuting_stopped() and s:
            """Turns off all commuting."""
            if not 'turn_off_Ts' in self.storage:
                self.storage['turn_off_Ts'] = []
            if not 'turn_off_cTs' in self.storage:
                self.storage['turn_off_cTs'] = []

            turn_off_Ts = self.storage['turn_off_Ts']
            turn_off_cTs = self.storage['turn_off_cTs']

            # Turn off commuting by setting t2=-1 for all transporters,
            # and store the t2 values so that commuting can be turned on again

            for i in range(self.Ts_num):
                turn_off_Ts.append( self.Ts[i].t2 )
                self.Ts[i].t2 = -1

            for i in range(self.cTs_num):
                turn_off_cTs.append( self.cTs[i].t2 )
                self.cTs[i].t2 = -1

            self.storage['commuting_is_stopped'] = True
        elif self.is_commuting_stopped():
            turn_off_Ts = self.storage['turn_off_Ts']
            turn_off_cTs = self.storage['turn_off_cTs']

            # Restore the t2 values of the transporters

            for i, t2 in zip(range(self.Ts_num), turn_off_Ts):
                self.Ts[i].t2 = t2

            for i, t2 in zip(range(self.cTs_num), turn_off_cTs):
                self.cTs[i].t2 = t2

            self.storage['commuting_is_stopped'] = False

    cpdef bint is_commuting_stopped(self):
        if 'commuting_is_stopped' in self.storage:
            return self.storage['commuting_is_stopped']
        else:
            return False



cdef void SIM_EVENT_NULL(csimulation cg, int step_i, DTYPE_t t, DTYPE_t dt, DTYPE_t[:] X_state, DTYPE_t[:] dX_state):
    raise(Exception('SIM_EVENT_NULL triggered'))

