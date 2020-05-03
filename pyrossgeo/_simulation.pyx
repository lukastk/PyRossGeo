from pyrossgeo.csimulation cimport DTYPE_t, SIM_EVENT, SIM_EVENT_NULL, csimulation, node, cnode, transporter
from pyrossgeo.csimulation import DTYPE

import numpy as np
cimport numpy as np
import zarr
from libc.math cimport exp
from cython.parallel import prange
import cython

def simulate(self, X_state, t_start, t_end, dts, steps_per_save=-1, out_file="", steps_per_print=-1, only_save_nodes=False,
                    event_times=[], event_function=None):
    return csimulate(self, X_state, t_start, t_end, dts, steps_per_save, out_file, steps_per_print, only_save_nodes, event_times, event_function)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef csimulate(csimulation self, DTYPE_t[:] X_state, DTYPE_t t_start, DTYPE_t t_end, object _dts, int steps_per_save=-1,
                            str out_file="", int steps_per_print=-1, bint only_save_nodes=False,
                            object event_times=[], object event_function=None,
                            object cevent_times=[], SIM_EVENT cevent_function=SIM_EVENT_NULL):

    #########################################################################
    #### Definitions ########################################################
    #########################################################################

    #### Forward-Euler variables ########################################

    cdef DTYPE_t t = t_start
    cdef DTYPE_t dt
    cdef DTYPE_t[:] dts
    cdef DTYPE_t tday
    cdef int dts_num
    cdef int step_i
    cdef int steps
    cdef np.ndarray X_state_arr = np.asarray(X_state)

    #### System variables ###############################################

    cdef int age_groups = self.age_groups
    cdef int model_dim = self.model_dim
    cdef int max_node_index = self.max_node_index
    cdef int X_state_size = len(X_state)
    cdef node* nodes = self.nodes
    cdef int nodes_num = self.nodes_num
    cdef cnode* cnodes = self.cnodes
    cdef int cnodes_num = self.cnodes_num
    cdef int state_size = self.state_size
    cdef int node_states_len = self.node_states_len

    # Model
    cdef int** class_infections = self.class_infections
    cdef int* class_infections_num = self.class_infections_num
    cdef int[:] infection_classes_indices = self.infection_classes_indices
    cdef int infection_classes_num = self.infection_classes_num
    cdef int** linear_terms = self.linear_terms
    cdef int* linear_terms_num = self.linear_terms_num
    cdef DTYPE_t[:,:,:] contact_matrices = self.contact_matrices
    cdef int[:, :] node_infection_cmats = self.node_infection_cmats
    cdef int[:, :] cnode_infection_cmats = self.cnode_infection_cmats

    # Transport
    cdef transporter* Ts = self.Ts # Going into commuterverses
    cdef int Ts_num = self.Ts_num
    cdef transporter* cTs = self.cTs # Going out from commuterverses
    cdef int cTs_num = self.cTs_num

    # Used for lambda calculation
    cdef int*** nodes_at_j = self.nodes_at_j
    cdef int** nodes_at_j_len = self.nodes_at_j_len

    # Used for tau calculation
    cdef int*** cnodes_into_k = self.cnodes_into_k
    cdef int** cnodes_into_k_len = self.cnodes_into_k_len
    
    # Transport profile
    cdef DTYPE_t transport_profile_integrated = self.transport_profile_integrated
    cdef DTYPE_t transport_profile_integrated_r = self.transport_profile_integrated_r
    cdef DTYPE_t transport_profile_m = self.transport_profile_m
    cdef DTYPE_t transport_profile_c = self.transport_profile_c
    cdef DTYPE_t transport_profile_c_r = self.transport_profile_c_r

    #### Simulation variables ###########################################

    cdef int cni, ni, si, Ti, cTi, i, j, ui, u, o, cmat_i, oi, X_index, loc_j, to_k, age_a, age_b
    cdef DTYPE_t S, t1, t2, transport_profile, fro_N, cn_N, mt, transport_profile_exponent
    cdef node n, fro_n, to_n
    cdef cnode cn
    
    cdef DTYPE_t[:, :] _lambdas = self._lambdas
    cdef DTYPE_t[:] _Is = self._Is
    cdef DTYPE_t[:] _Ns = self._Ns

    cdef int minutes_in_day = 1440

    cdef np.ndarray dX_state_arr
    cdef DTYPE_t[:] dX_state

    #### Calcuate steps for the Forward-Euler integration ###############

    if not type(_dts) == np.ndarray:
        _dts = np.array([_dts], dtype=DTYPE)
    dts = _dts

    t = t_start
    steps = 0
    while t <= t_end:
        t += dts[steps % len(dts)]
        steps += 1
    steps -= 1

    dX_state_arr = np.zeros( X_state.size )
    dX_state = dX_state_arr

    dts_num = len(dts)

    #### Set-up variables for storing the simulation history ############

    cdef int save_i, X_states_saved_col_num
    X_states_saved = None
    ts_saved = None

    if steps_per_save != -1:

        # Compute the number of save states
        t = t_start
        save_states = 1
        for step_i in range(steps):
            t += dts[steps % len(dts)]
            if step_i % steps_per_save == 0:
                save_states += 1

        if only_save_nodes:
            X_states_saved_col_num = self.node_states_len
        else:
            X_states_saved_col_num = X_state.size

        # If out_file is not specified, then the states will be stored to numpy array
        if out_file == "":
            X_states_saved = np.zeros( ( save_states, X_states_saved_col_num) )

        # If out_file is specified, then the states will be stored on the harddrive directly, using a zarr array
        else: 
            X_states_saved = zarr.open('%s.zarr' % out_file, mode='w',
                                shape=( save_states, X_states_saved_col_num ),
                                chunks=(1 , X_states_saved_col_num), dtype=DTYPE)
            
        X_states_saved[0, :] = X_state_arr[:X_states_saved_col_num]
        ts_saved = np.zeros( X_states_saved.shape[0] )
        ts_saved[0] = t_start
        save_i = 1

    #### Event management ###############################################

    cdef np.ndarray event_steps_arr = np.full(steps, 0, dtype=np.int8)
    cdef char[:] event_steps = event_steps_arr

    cdef np.ndarray cevent_steps_arr = np.full(steps, 0, dtype=np.int8)
    cdef char[:] cevent_steps = cevent_steps_arr

    t = t_start
    for step_i in range(steps):
        dt = dts[step_i % dts_num]
        t += dt

        for et in event_times:
            if et <= t and t < et+dt:
                event_steps[step_i] = 1

        for et in cevent_times:
            if et <= t and t < et+dt:
                cevent_steps[step_i] = 1

    #########################################################################
    #### Simulation #########################################################
    #########################################################################

    t = t_start

    for step_i in range(steps):
        tday = t % minutes_in_day
        dt = dts[step_i % dts_num]

        # Reset dX_state to 0
        for i in prange(state_size, nogil=True):
            dX_state[i] = 0

        #####################################################################
        #### Dynamics #######################################################
        #####################################################################

        #### Node dynamics ##################################################

        for loc_j in range(max_node_index+1):

            # Find the populations of each age group

            for age_a in range(age_groups):
                _Ns[age_a] = 0
                for i in range(nodes_at_j_len[age_a][loc_j]):
                    n = nodes[nodes_at_j[age_a][loc_j][i]]
                    for o in range(model_dim):#prange(model_dim, nogil=True):
                        _Ns[age_a] += X_state[n.state_index + o]

            # Compute lambdas

            for ui in range(infection_classes_num):
                u = infection_classes_indices[ui]
                cmat_i = node_infection_cmats[loc_j][u]

                # Find the infecteds of each age group

                for age_a in range(age_groups):
                    _Is[age_a] = 0
                    for i in range(nodes_at_j_len[age_a][loc_j]):#prange(nodes_at_j_len[age_a][loc_j], nogil=True):
                        n = nodes[nodes_at_j[age_a][loc_j][i]]
                        _Is[age_a] += X_state[n.state_index + u]

                # Compute lambdas

                for age_a in range(age_groups):
                    _lambdas[age_a][ui] = 0
                    for age_b in range(age_groups):#prange(age_groups, nogil=True):
                        if _Ns[age_b] > 1: # No infections can occur if there are fewer than one person at node
                            _lambdas[age_a][ui] += contact_matrices[cmat_i][age_a][age_b] * _Is[age_b] / _Ns[age_b]

            # Apply to each node
            
            for age_a in range(age_groups):
                for i in range(nodes_at_j_len[age_a][loc_j]): 
                    n = nodes[nodes_at_j[age_a][loc_j][i]]
                    si = n.state_index
                    S = X_state[si]

                    for o in range(model_dim):
                        # Apply infection terms
                        for j in range(class_infections_num[o]):#prange(class_infections_num[o], nogil=True):
                            ui = class_infections[o][j]
                            dX_state[si+o] += n.infection_coeffs[o][j] * _lambdas[age_a][ui] * S

                        # Apply linear terms
                        for j in range(linear_terms_num[o]):#prange(linear_terms_num[o], nogil=True):
                            u = linear_terms[o][j]
                            if X_state[ si + u ] > 0: # Only allow interaction if the class is positive
                                dX_state[si+o] += n.linear_coeffs[o][j] * X_state[ si + u ]

        #### CNode dynamics #################################################

        for to_k in range(max_node_index+1):

            # Find the populations of each age group

            for age_a in range(age_groups):
                _Ns[age_a] = 0
                for i in range(cnodes_into_k_len[age_a][to_k]):
                    cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                    for o in range(model_dim):#prange(model_dim, nogil=True):
                        _Ns[age_a] += X_state[cn.state_index + o]

            # Compute lambdas

            for ui in range(infection_classes_num):
                u = infection_classes_indices[ui]
                cmat_i = cnode_infection_cmats[to_k][u]

                # Find the infecteds of each age group

                for age_a in range(age_groups):
                    _Is[age_a] = 0
                    for i in range(cnodes_into_k_len[age_a][to_k]):#prange(cnodes_into_k_len[age_a][to_k], nogil=True):
                        cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                        _Is[age_a] += X_state[cn.state_index + u]

                # Compute lambdas

                for age_a in range(age_groups):
                    _lambdas[age_a][ui] = 0
                    for age_b in range(age_groups):#prange(age_groups, nogil=True):
                        if _Ns[age_b] > 1: # No infections can occur if there are fewer than one person at node
                            _lambdas[age_a][ui] += contact_matrices[cmat_i][age_a][age_b] * _Is[age_b] / _Ns[age_b]

            # Apply to each node

            for age_a in range(age_groups):
                for i in range(cnodes_into_k_len[age_a][to_k]):
                    cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                    si = cn.state_index
                    S = X_state[si]

                    for o in range(model_dim):#prange(model_dim):
                        # Apply infection terms
                        for j in range(class_infections_num[o]):#prange(class_infections_num[o], nogil=True):
                            ui = class_infections[o][j]
                            dX_state[si+o] += cn.infection_coeffs[o][j] * _lambdas[age_a][ui] * S
                        
                        # Apply linear terms
                        for j in range(linear_terms_num[o]):#prange(linear_terms_num[o], nogil=True):
                            u = linear_terms[o][j]
                            if X_state[ si + u ] > 0: # Only allow interaction if the class is positive
                                dX_state[si+o] += cn.linear_coeffs[o][j] * X_state[ si + u ]


        #####################################################################
        #### Transport ######################################################
        #####################################################################

        #### Node to CNode ##################################################

        for Ti in range(Ts_num):

            t1 = Ts[Ti].t1
            t2 = Ts[Ti].t2

            if tday >= t1 and tday <= t2:
                fro_n = nodes[Ts[Ti].fro_node_index]
                cn = cnodes[Ts[Ti].cnode_index]

                fro_N = 0
                for oi in range(model_dim):#prange(model_dim, nogil=True):
                    fro_N += X_state[fro_n.state_index + oi]

                if not Ts[Ti].is_on:
                    if Ts[Ti].use_percentage:
                        Ts[Ti].N0 = fro_N*Ts[Ti].move_percentage
                    else:
                        Ts[Ti].N0 = Ts[Ti].move_N
                    Ts[Ti].is_on = True

                transport_profile_exponent = (tday - t1)/(t2-t1) - transport_profile_m
                transport_profile = exp(- transport_profile_exponent * transport_profile_exponent * transport_profile_c_r) * transport_profile_integrated_r * Ts[Ti].r_T_Delta_t
                
                if fro_N <= 0:
                    continue
                
                si = fro_n.state_index
                for oi in range(model_dim):#prange(model_dim, nogil=True):
                    if not Ts[Ti].moving_classes[oi]:
                        continue

                    mt = Ts[Ti].N0 * transport_profile * (X_state[fro_n.state_index+oi] / fro_N)
                    if X_state[si+oi] + dt*(dX_state[si+oi] - mt) < 0:
                        mt = X_state[si+oi]/dt
                        dX_state[cn.state_index+oi] += mt + dX_state[si+oi] # We shift the SIR dynamics that transpired in the node into the cnode
                        dX_state[si+oi] += -(mt + dX_state[si+oi])
                    else:
                        dX_state[si+oi] -= mt
                        dX_state[cn.state_index+oi] += mt

            else:
                Ts[Ti].is_on = False
        
        #### CNode to Node ##################################################
        
        for cTi in range(cTs_num):

            t1 = cTs[cTi].t1
            t2 = cTs[cTi].t2

            if tday >= t1 and tday <= t2:
                cn = cnodes[cTs[cTi].cnode_index]
                to_node = nodes[cTs[cTi].to_node_index]

                cn_N = 0
                for oi in range(model_dim):#prange(model_dim, nogil=True):
                    cn_N += X_state[cn.state_index + oi]

                if not cTs[cTi].is_on:
                    if cTs[cTi].use_percentage:
                        cTs[cTi].N0 = cn_N*cTs[cTi].move_percentage
                    else:
                        cTs[cTi].N0 = cTs[cTi].move_N
                    cTs[cTi].is_on = True

                transport_profile_exponent = (tday - t1)/(t2-t1) - transport_profile_m
                transport_profile = exp(- transport_profile_exponent * transport_profile_exponent * transport_profile_c_r) * transport_profile_integrated_r * Ts[Ti].r_T_Delta_t

                if cn_N <= 0:
                    continue

                si = cn.state_index
                for oi in range(model_dim):#prange(model_dim, nogil=True):
                    if not cTs[cTi].moving_classes[oi]:
                        continue

                    if tday+dt >= t2: # If the commuting window is ending, all must leave the commuterverse
                        mt = X_state[si+oi]/dt
                        dX_state[to_node.state_index+oi] += mt + dX_state[si+oi] # We shift the SIR dynamics that transpired in the cnode into the node
                        dX_state[si+oi] += - (mt + dX_state[si+oi])
                    else:
                        mt = cTs[cTi].N0 * transport_profile * (X_state[si+oi] / cn_N)
                        if X_state[si+oi] + dt*(dX_state[si+oi] - mt) < 0:
                            mt = X_state[si+oi]/dt
                            dX_state[to_node.state_index+oi] += mt + dX_state[si+oi] # We shift the SIR dynamics that transpired in the cnode into the node
                            dX_state[si+oi] += - (mt + dX_state[si+oi])
                        else:
                            dX_state[si+oi] -= mt
                            dX_state[to_node.state_index+oi] += mt
            else:
                cTs[cTi].is_on = False

        #####################################################################
        #### Forward-Euler ##################################################
        #####################################################################

        for j in prange(X_state_size, nogil=True):
            X_state[j] += dX_state[j]*dt

        t += dt

        if steps_per_print != -1 and step_i % steps_per_print==0:
            print("Step %s out of %s" % (step_i, steps))

        #### Store state

        if steps_per_save != -1:

            if (step_i+1) % steps_per_save == 0:
                X_states_saved[save_i,:] = X_state_arr[:X_states_saved_col_num]
                ts_saved[save_i] = t
                save_i += 1

        #### Call event function

        if event_steps[step_i]:
            event_function(self, step_i, t, dt, X_state, dX_state)

        #### Call Cython event function

        if cevent_steps[step_i]:
            cevent_function(self, step_i, t, dt, X_state, dX_state)

    if steps_per_save != -1:
        sim_data = (self.state_mappings, ts_saved, X_states_saved)

        return sim_data