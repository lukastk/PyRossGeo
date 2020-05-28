import zarr
import pickle
import cython
cimport libc.math
from libc.stdlib cimport malloc, free
from cython.parallel import prange
import numpy as np
cimport numpy as np
import time # For seeding random
import scipy.stats

from pyrossgeo.__defs__ import DTYPE, contact_scaling_types

#@cython.wraparound(False)
#@cython.boundscheck(True)
#@cython.cdivision(False)
#@cython.nonecheck(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef simulate(Simulation self, DTYPE_t[:] X_state, DTYPE_t t_start, DTYPE_t t_end, object _dts, int steps_per_save=-1,
                            str save_path="", bint only_save_nodes=False, int steps_per_print=-1,
                            int random_seed=-1):

    if not self.has_been_initialized:
        raise Exception("Must initialise before starting simulation.")
    
    ####################################################################
    #### Definitions ###################################################
    ####################################################################

    #### Forward-Euler variables #######################################
    
    cdef DTYPE_t t = t_start
    cdef DTYPE_t dt
    cdef DTYPE_t r_dt
    cdef DTYPE_t[:] dts
    cdef DTYPE_t tday
    cdef int dts_num
    cdef int step_i
    cdef int steps
    cdef np.ndarray X_state_arr = np.asarray(X_state)

    #### System variables ##############################################

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

    cdef np.ndarray total_N_arr = np.zeros( age_groups )
    cdef np.ndarray Ns_arr = np.zeros( (max_node_index+1, age_groups) )
    cdef np.ndarray cNs_arr = np.zeros( (max_node_index+1, age_groups) )
    cdef np.ndarray Os_arr = np.zeros( (max_node_index+1, age_groups, model_dim) )
    cdef np.ndarray cOs_arr = np.zeros( (max_node_index+1, age_groups, model_dim) )
    
    cdef DTYPE_t[:] total_N = total_N_arr
    cdef DTYPE_t[:,:] Ns = Ns_arr
    cdef DTYPE_t[:,:] cNs = cNs_arr
    cdef DTYPE_t[:,:,:] Os = Os_arr
    cdef DTYPE_t[:,:,:] cOs = cOs_arr

    # Model
    cdef model_term* model_linear_terms = self.model_linear_terms
    cdef int model_linear_terms_len = self.model_linear_terms_len
    cdef model_term* model_infection_terms = self.model_infection_terms
    cdef int model_infection_terms_len = self.model_infection_terms_len
    cdef int[:] infection_classes_indices = self.infection_classes_indices
    cdef int infection_classes_num = self.infection_classes_num

    cdef DTYPE_t[:,:,:] contact_matrices = self.contact_matrices
    cdef int contact_matrices_num = self.contact_matrices.shape[0]
    cdef int** contact_matrices_at_each_loc = self.contact_matrices_at_each_loc
    cdef int* contact_matrices_at_each_loc_len = self.contact_matrices_at_each_loc_len
    cdef int** contact_matrices_at_each_to = self.contact_matrices_at_each_to
    cdef int* contact_matrices_at_each_to_len = self.contact_matrices_at_each_to_len
    
    cdef int max_contact_matrices_used_at_single_node = np.max( np.concatenate([
        [contact_matrices_at_each_loc_len[i] for i in range(max_node_index+1)],
        [contact_matrices_at_each_to_len[i] for i in range(max_node_index+1)]
    ])) # Used to create _lambda

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

    # Stochasticity
    cdef bint stochastic_simulation = self.stochastic_simulation
    total_Os_arr = np.zeros(model_dim, dtype=DTYPE)
    cdef DTYPE_t[:] total_Os = total_Os_arr # Used to see whether stochasticity should be turned on
    cdef DTYPE_t[:] stochastic_threshold_from_below = self.stochastic_threshold_from_below
    cdef DTYPE_t[:] stochastic_threshold_from_above = self.stochastic_threshold_from_above
    cdef bint* loc_j_is_stochastic
    cdef bint* to_k_is_stochastic

    # Contact matrix scaling
    cdef int contact_scaling_type = self.contact_scaling_type
    cdef DTYPE_t[:] contact_scaling_params = self.contact_scaling_params
    cdef DTYPE_t _f, _g, _cg
    
    location_r_area_arr = np.array([1.0/a if not np.isnan(a) else np.nan for a in self.location_area]) # Compute inverse of area
    cdef DTYPE_t[:] location_r_area = location_r_area_arr

    commuterverse_r_area_arr = np.array([1.0/a if not np.isnan(a) else np.nan for a in self.commuterverse_area])
    cdef DTYPE_t[:] commuterverse_r_area = commuterverse_r_area_arr

    cdef np.ndarray cmat_scaling_fg_arr = np.zeros( ( self.max_node_index+1, age_groups, age_groups) )
    cdef np.ndarray cmat_scaling_fg_cverse_arr = np.zeros( ( self.max_node_index+1, age_groups, age_groups) )
    cdef np.ndarray cmat_scaling_a_arr = np.zeros( (age_groups, age_groups) )
    cdef DTYPE_t[:,:,:] cmat_scaling_fg = cmat_scaling_fg_arr
    cdef DTYPE_t[:,:,:] cmat_scaling_fg_cverse = cmat_scaling_fg_cverse_arr
    cdef DTYPE_t[:,:] cmat_scaling_a = cmat_scaling_a_arr
     
    if random_seed == -1:
        random_seed = np.int64(np.round(time.time()))
    cdef mt19937 gen = mt19937(random_seed)
    cdef poisson_distribution[int] dist

    #if random_seed == -1:
    #    np.random.seed( np.int64(np.round(time.time())) )

    # Consts
    cdef int minutes_in_day = 1440
    save_node_mappings_path = 'node_mappings.pkl'
    save_cnode_mappings_path = 'cnode_mappings.pkl'
    save_ts_path = 'ts.npy'
    save_X_states_path = 'X_states.zarr'

    #### Simulation variables ##########################################

    cdef int cni, ni, si, Ti, cTi, i, j, ui, u, o, cmat_i, oi, X_index, loc_j, to_k, age_a, age_b, loc
    cdef bint to_k_is_active = False
    cdef DTYPE_t S, t1, t2, transport_profile, fro_N, cn_N, term, transport_profile_exponent, _N
    cdef node n, fro_n, to_n
    cdef cnode cn
    cdef model_term mt
    
    _lambdas_arr = np.zeros(( max_contact_matrices_used_at_single_node, self.age_groups, self.infection_classes_num ))
    cdef DTYPE_t[:, :, :] _lambdas = _lambdas_arr
    cdef DTYPE_t[:] _Is = self._Is
    cdef DTYPE_t[:] _Ns = self._Ns

    cdef np.ndarray dX_state_arr
    cdef DTYPE_t[:] dX_state

    #### Compute total population ######################################

    for i in range(nodes_num):
        n = self.nodes[i]
        for o in range(model_dim):
            total_N[n.age] += X_state[n.state_index + o]

    for i in range(cnodes_num):
        cn = self.cnodes[i]
        for o in range(model_dim):
            total_N[cn.age] += X_state[cn.state_index + o]

    #### Initialize stochasticity ######################################

    loc_j_is_stochastic = <bint *> malloc( (self.max_node_index+1) * sizeof(bint))
    to_k_is_stochastic = <bint *> malloc( (self.max_node_index+1) * sizeof(bint))

    if stochastic_simulation:

        for loc_j in range(max_node_index+1):
            n = nodes[i]

            for o in range(model_dim):
                total_Os[o] = 0

            for age_a in range(age_groups):
                for i in range(nodes_at_j_len[age_a][loc_j]):
                    n = nodes[nodes_at_j[age_a][loc_j][i]]
                    for o in range(model_dim):
                        total_Os[o] += X_state[n.state_index + o]

            loc_j_is_stochastic[loc_j] = False

            for o in range(model_dim):
                loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stochastic_threshold_from_below[o])

        for to_k in range(max_node_index+1):
            n = nodes[i]

            for o in range(model_dim):
                total_Os[o] = 0

            for age_a in range(age_groups):
                for i in range(cnodes_into_k_len[age_a][to_k]):
                    cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                    for o in range(model_dim):
                        total_Os[o] += X_state[n.state_index + o]

            to_k_is_stochastic[to_k] = False

            for o in range(model_dim):
                to_k_is_stochastic[to_k] = to_k_is_stochastic[to_k] or (total_Os[o] < stochastic_threshold_from_below[o])

    # Loop through nodes and cnodes to see if they should start out as stochastic or not

    #### Calcuate steps for the Forward-Euler integration ##############

    if type(_dts) == list:
        _dts= np.array(_dts)
    elif not type(_dts) == np.ndarray:
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

    #### Set-up variables for storing the simulation history ###########

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

        # If save_path is not specified, then the states will be stored to numpy array
        if save_path == "":
            X_states_saved = np.zeros( ( save_states, X_states_saved_col_num) )

        # If save_path is specified, then the states will be stored on the harddrive directly, using a zarr array
        else: 
            X_states_saved = zarr.open('%s/%s' % (save_path, save_X_states_path), mode='w',
                                shape=( save_states, X_states_saved_col_num ),
                                chunks=(1 , X_states_saved_col_num), dtype=DTYPE)
            
        X_states_saved[0, :] = X_state_arr[:X_states_saved_col_num]
        ts_saved = np.zeros( X_states_saved.shape[0] )
        ts_saved[0] = t_start
        save_i = 1

    #### Event management ##############################################

    cdef list event_functions = self.event_functions
    cdef np.ndarray is_event_on_arr = np.zeros((steps, len(self.event_functions)), dtype=np.int8)
    cdef char[:,:] is_event_on = is_event_on_arr
    cdef int event_i

    is_a_event_on_arr = np.zeros(steps, dtype=np.int8)
    cdef char[:] is_a_event_on = is_a_event_on_arr

    # Align event times with simulation steps

    t = t_start
    for step_i in range(steps):
        dt = dts[step_i % dts_num]
        t += dt

        for event_i in range(len(self.event_times)):
            for et in self.event_times[event_i]:
                if self.event_repeat_times[event_i] == -1:
                    _t = t
                else:
                    _t = (t % self.event_repeat_times[event_i])

                if et <= _t and _t < et+dt:
                    is_event_on_arr[step_i][event_i] = 1
                    is_a_event_on[step_i] = 1

    ####################################################################
    #### Simulation ####################################################
    ####################################################################

    t = t_start

    for step_i in range(steps):
        tday = t % minutes_in_day
        dt = dts[step_i % dts_num]
        r_dt = 1/dt

        # Reset dX_state to 0
        for i in prange(state_size, nogil=True):
            dX_state[i] = 0

        ################################################################
        #### Dynamics ##################################################
        ################################################################

        #### Contact matrix scaling ####################################

        for i in range(age_groups):
            for j in range(age_groups):
                cmat_scaling_a[i,j] = 0

        for loc in range(max_node_index+1):

            # Reset arrays
            for age in range(age_groups):
                Ns_arr[loc,age] = 0
                cNs_arr[loc,age] = 0
                for o in range(model_dim):
                    Os[loc,age,o] = 0
                    cOs[loc,age,o] = 0

            # Nodes
            for age in range(age_groups):
                for i in range(nodes_at_j_len[age][loc]):
                    n = nodes[nodes_at_j[age][loc][i]]

                    _N = 0
                    for o in range(model_dim):
                        Os[loc,age,o] += X_state[n.state_index + o]
                        _N += X_state[n.state_index + o]

                    if _N < 1e-1:
                        n.is_on = False
                    else:
                        n.is_on = True

                    Ns[loc,age] += _N

            # Cnodes
            for age in range(age_groups):
                for i in range(cnodes_into_k_len[age][loc]):
                    cn = cnodes[cnodes_into_k[age][loc][i]]

                    if cn.is_on:
                        _N = 0
                        for o in range(model_dim):
                            cOs[loc,age,o] += X_state[cn.state_index + o]
                            _N += X_state[cn.state_index + o]

                        cNs[loc,age] += _N

            # Compute scaling tensor

            for i in range(age_groups):
                for j in range(age_groups):

                    _g = Ns[loc, i]*Ns[loc, j] * location_r_area[loc]*location_r_area[loc]
                    _cg = cNs[loc, i]*cNs[loc, j] * commuterverse_r_area[loc]*commuterverse_r_area[loc]

                    if contact_scaling_type == contact_scaling_types.linear:
                        _g = contact_scaling_linear(_g, contact_scaling_params)
                        _cg = contact_scaling_linear(_cg, contact_scaling_params)
                    elif contact_scaling_type == contact_scaling_types.powerlaw:
                        _g = contact_scaling_powerlaw(_g, contact_scaling_params)
                        _cg = contact_scaling_powerlaw(_cg, contact_scaling_params)
                    elif contact_scaling_type == contact_scaling_types.exp:
                        _g = contact_scaling_exp(_g, contact_scaling_params)
                        _cg = contact_scaling_exp(_cg, contact_scaling_params)
                    elif contact_scaling_type == contact_scaling_types.log:
                        _g = contact_scaling_log(_g, contact_scaling_params)
                        _cg = contact_scaling_log(_cg, contact_scaling_params)
                    
                    _f = libc.math.sqrt( total_N[i] * Ns[loc, j] / (Ns[loc, i] * total_N[j]) )
                    cmat_scaling_fg[loc,i,j] = _f*_g
                    cmat_scaling_a[i,j] += Ns[loc, i]*cmat_scaling_fg[loc,i,j]

                    if cNs[loc, i] != 0:
                        _f = libc.math.sqrt( total_N[i] * cNs[loc, j] / (cNs[loc, i] * total_N[j]) )
                        cmat_scaling_fg_cverse[loc,i,j] = _f*_cg
                        cmat_scaling_a[i,j] += cNs[loc, i]*cmat_scaling_fg_cverse[loc,i,j]
                    else:
                        cmat_scaling_fg_cverse[loc,i,j]=0

        for i in range(age_groups):
            for j in range(age_groups):
                cmat_scaling_a[i,j] = cmat_scaling_a[i,j] / total_N[i]

        # Normalize cmat_scaling_fg and cmat_scaling_fg_cverse using cmat_scaling_a

        for loc in range(max_node_index+1):
            for i in range(age_groups):
                for j in range(age_groups):
                    cmat_scaling_fg[loc,i,j] = cmat_scaling_fg[loc,i,j] / cmat_scaling_a[i,j]
                    cmat_scaling_fg_cverse[loc,i,j] = cmat_scaling_fg_cverse[loc,i,j] / cmat_scaling_a[i,j]

        #### Node dynamics #############################################

        for loc_j in range(max_node_index+1):

            #### Compute lambdas

            for ui in range(infection_classes_num):
                u = infection_classes_indices[ui]

                # Compute lambdas

                for cmat_i in range(contact_matrices_at_each_loc_len[loc_j]):
                    for age_a in range(age_groups):
                        _lambdas[cmat_i][age_a][ui] = 0
                        for age_b in range(age_groups):
                            if Ns[loc_j,age_b] > 1: # No infections can occur if there are fewer than one person at node
                                _lambdas[cmat_i][age_a][ui] += cmat_scaling_fg[loc_j][age_a][age_b]*contact_matrices[contact_matrices_at_each_loc[loc_j][cmat_i]][age_a][age_b] * Os[loc_j,age_b,u] / Ns[loc_j,age_b]

            #### Decide whether deterministic or stochastic

            for o in range(model_dim):
                total_Os[o] = 0
                for age in range(age_groups):
                    total_Os[o] += Os[loc_j,age,o]
            
            if stochastic_simulation:
                if loc_j_is_stochastic[loc_j]:
                    loc_j_is_stochastic[loc_j] = False
                    for o in range(model_dim):
                        loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stochastic_threshold_from_below[o])
                else:
                    loc_j_is_stochastic[loc_j] = False
                    for o in range(model_dim):
                        loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stochastic_threshold_from_above[o])
            
            #### Compute the derivatives at each node

            # Stochastic

            if stochastic_simulation and loc_j_is_stochastic[loc_j]:
                for age_a in range(age_groups):
                    for i in range(nodes_at_j_len[age_a][loc_j]): 
                        n = nodes[nodes_at_j[age_a][loc_j][i]]

                        if not n.is_on:
                            continue

                        si = n.state_index
                        S = X_state[si] # S is always located at the state index

                        for j in range(model_linear_terms_len):
                            mt = model_linear_terms[j]   
                            if X_state[si+mt.oi_neg] > 0: # Only allow interaction if the class is positive
                                dist = poisson_distribution[int](dt*n.linear_coeffs[j]*X_state[si+mt.oi_coupling])
                                term = dist(gen) * r_dt
                                #term = scipy.stats.poisson.rvs(dt*n.linear_coeffs[j]*X_state[si+mt.oi_coupling]) * r_dt
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

                        for j in range(model_infection_terms_len):
                            mt = model_infection_terms[j]
                            if X_state[si+mt.oi_neg] > 0: # Only allow interaction if the class is positive
                                cmat_i = n.contact_matrix_indices[mt.infection_index]
                                dist = poisson_distribution[int](dt*n.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S)
                                term = dist(gen) * r_dt
                                #term = scipy.stats.poisson.rvs(dt*n.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S) * r_dt

                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term      
 
            # Deterministic
            else:
                for age_a in range(age_groups):
                    for i in range(nodes_at_j_len[age_a][loc_j]): 
                        n = nodes[nodes_at_j[age_a][loc_j][i]]

                        if not n.is_on:
                            continue

                        si = n.state_index
                        S = X_state[si] # S is always located at the state index

                        for j in range(model_linear_terms_len):
                            mt = model_linear_terms[j]
                            if X_state[si+mt.oi_neg] > 0: # Only allow interaction if the class is positive
                                term = n.linear_coeffs[j] * X_state[si+mt.oi_coupling]
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

                        for j in range(model_infection_terms_len):
                            mt = model_infection_terms[j]

                            if X_state[si+mt.oi_neg] > 0: # Only allow interaction if the class is positive
                                cmat_i = n.contact_matrix_indices[mt.infection_index]
                                term = n.infection_coeffs[j] * _lambdas[cmat_i][age_a][mt.infection_index] * S
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

        #### CNode dynamics ############################################

        for to_k in range(max_node_index+1):

            # Check whether there are any active cnodes going into k

            to_k_is_active = False
            for age_a in range(age_groups):
                for i in range(cnodes_into_k_len[age_a][to_k]):
                    cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                    to_k_is_active = cn.is_on or to_k_is_active
                    if to_k_is_active:
                        break
                if to_k_is_active:
                    break

            if not to_k_is_active:
                continue

            #### Compute lambdas

            for ui in range(infection_classes_num):
                u = infection_classes_indices[ui]

                # Compute lambdas

                for cmat_i in range(contact_matrices_at_each_to_len[to_k]):
                    for age_a in range(age_groups):
                        _lambdas[cmat_i][age_a][ui] = 0
                        for age_b in range(age_groups):
                            if cNs[to_k,age_b] > 1: # No infections can occur if there are fewer than one person at node
                                _lambdas[cmat_i][age_a][ui] += cmat_scaling_fg_cverse[to_k][age_a][age_b]*contact_matrices[contact_matrices_at_each_to[to_k][cmat_i]][age_a][age_b] * Os[to_k,age_b,u] / cNs[to_k,age_b]

            #### Decide whether deterministic or stochastic

            for o in range(model_dim):
                total_Os[o] = 0
                for age in range(age_groups):
                    total_Os[o] += Os[to_k,age,o]

            if stochastic_simulation:
                if to_k_is_stochastic[to_k]:
                    to_k_is_stochastic[to_k] = False
                    for o in range(model_dim):
                        to_k_is_stochastic[to_k] = to_k_is_stochastic[to_k] or (total_Os[o] < stochastic_threshold_from_below[o])
                else:
                    to_k_is_stochastic[to_k] = True
                    for o in range(model_dim):
                        to_k_is_stochastic[to_k] = to_k_is_stochastic[to_k] and (total_Os[o] > stochastic_threshold_from_above[o])

            # Stochastic
            if stochastic_simulation and to_k_is_stochastic[to_k]:
                for age_a in range(age_groups):
                    for i in range(cnodes_into_k_len[age_a][to_k]): 
                        cn = cnodes[cnodes_into_k[age_a][to_k][i]]

                        if not cn.is_on:
                            continue

                        si = cn.state_index
                        S = X_state[si] # S is always located at the state index

                        for j in range(model_linear_terms_len):
                            mt = model_linear_terms[j]
                            if X_state[si+mt.oi_neg] > 0: # Only allow interaction if the class is positive
                                #dist = poisson_distribution[int](dt*cn.linear_coeffs[j]*X_state[si+mt.oi_coupling])
                                #term = dist(gen) * r_dt
                                term = scipy.stats.poisson.rvs(dt*cn.linear_coeffs[j]*X_state[si+mt.oi_coupling]) * r_dt
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

                        for j in range(model_infection_terms_len):
                            mt = model_infection_terms[j]
                            if X_state[si+mt.oi_neg] > 0: # Only allow interaction if the class is positive
                                cmat_i = cn.contact_matrix_indices[mt.infection_index]
                                #dist = poisson_distribution[int](dt*cn.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S)
                                #term = dist(gen) * r_dt
                                term = scipy.stats.poisson.rvs(dt*cn.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S) * r_dt
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term
            # Deterministic
            else:
                for age_a in range(age_groups):
                    for i in range(cnodes_into_k_len[age_a][to_k]): 
                        cn = cnodes[cnodes_into_k[age_a][to_k][i]]

                        if not cn.is_on:
                            continue

                        si = cn.state_index
                        S = X_state[si] # S is always located at the state index

                        for j in range(model_linear_terms_len):
                            mt = model_linear_terms[j]
                            if X_state[si+mt.oi_neg] > 0: # Only allow interaction if the class is positive
                                term = cn.linear_coeffs[j] * X_state[si+mt.oi_coupling]
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

                        for j in range(model_infection_terms_len):
                            mt = model_infection_terms[j]
                            if X_state[si+mt.oi_neg] > 0: # Only allow interaction if the class is positive
                                cmat_i = cn.contact_matrix_indices[mt.infection_index]
                                term = cn.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term               
        
        ################################################################
        #### Transport #################################################
        ################################################################

        #### Node to CNode #############################################

        for Ti in range(Ts_num):

            t1 = Ts[Ti].t1
            t2 = Ts[Ti].t2

            if tday >= t1 and tday <= t2:
                fro_n = nodes[Ts[Ti].fro_node_index] # Origin node
                cn = cnodes[Ts[Ti].cnode_index] # Commuting node

                # Compute current population at origin node
                fro_N = 0
                for oi in range(model_dim):
                    fro_N += X_state[fro_n.state_index + oi]

                # If this commuting schedule is just starting, then
                # compute the number of people to move.
                if not Ts[Ti].is_on:
                    if Ts[Ti].use_percentage:
                        Ts[Ti].N0 = fro_N*Ts[Ti].move_percentage
                    else:
                        Ts[Ti].N0 = Ts[Ti].move_N
                    Ts[Ti].is_on = True
                    cn.is_on = True # Turn on commuter node. It will be turned off in the "CNode to Node" section

                # Compute the transport profile
                transport_profile_exponent = (tday - t1)*Ts[Ti].r_T_Delta_t - transport_profile_m
                transport_profile = libc.math.exp(- transport_profile_exponent * transport_profile_exponent * transport_profile_c_r) * transport_profile_integrated_r * Ts[Ti].r_T_Delta_t
                
                if fro_N <= 0:
                    continue
                
                si = fro_n.state_index
                for oi in range(model_dim):
                    if not Ts[Ti].moving_classes[oi]:
                        continue

                    # Compute the amount of people to move
                    term = Ts[Ti].N0 * transport_profile * (X_state[fro_n.state_index+oi] / fro_N)

                    # If the change will cause X_state[si+oi] to go negative,
                    # then adjust term so that X_state[si+oi] will be set 
                    # to 0.
                    if X_state[si+oi] + dt*(dX_state[si+oi] - term) < 0:
                        term = X_state[si+oi]*r_dt
                        dX_state[cn.state_index+oi] += term + dX_state[si+oi] # We shift the SIR dynamics that transpired in the node into the cnode
                        dX_state[si+oi] += -(term + dX_state[si+oi])
                    # Otherwise apply the transport as usual
                    else:
                        dX_state[si+oi] -= term
                        dX_state[cn.state_index+oi] += term

            else:
                Ts[Ti].is_on = False
        
        #### CNode to Node #############################################
        
        for cTi in range(cTs_num):

            t1 = cTs[cTi].t1
            t2 = cTs[cTi].t2

            if tday >= t1 and tday <= t2:
                cn = cnodes[cTs[cTi].cnode_index] # Commuting node
                to_node = nodes[cTs[cTi].to_node_index] # Destination node

                # Compute current population at the commuter node
                cn_N = 0
                for oi in range(model_dim):#prange(model_dim, nogil=True):
                    cn_N += X_state[cn.state_index + oi]

                # If this commuting schedule is just starting, then
                # compute the number of people to move.
                if not cTs[cTi].is_on:
                    if cTs[cTi].use_percentage:
                        cTs[cTi].N0 = cn_N*cTs[cTi].move_percentage
                    else:
                        cTs[cTi].N0 = cTs[cTi].move_N
                    cTs[cTi].is_on = True

                # Compute the transport profile
                transport_profile_exponent = (tday - t1)*cTs[cTi].r_T_Delta_t - transport_profile_m
                transport_profile = libc.math.exp(- transport_profile_exponent * transport_profile_exponent * transport_profile_c_r) * transport_profile_integrated_r * cTs[cTi].r_T_Delta_t

                if cn_N <= 0:
                    continue

                si = cn.state_index
                for oi in range(model_dim):#prange(model_dim, nogil=True):
                    if not cTs[cTi].moving_classes[oi]:
                        continue

                    # If the commuting window is ending, force all to leave the commuterverse
                    if tday+dt >= t2:
                        term = X_state[si+oi]*r_dt
                        dX_state[to_node.state_index+oi] += term + dX_state[si+oi] # We shift the SIR dynamics that transpired in the cnode into the node
                        dX_state[si+oi] += - (term + dX_state[si+oi])
                    else:
                        # Compute the amount of people to move
                        term = cTs[cTi].N0 * transport_profile * (X_state[si+oi] / cn_N)

                        # If the change will cause X_state[si+oi] to go negative,
                        # then adjust term so that X_state[si+oi] will be set 
                        # to 0. 
                        if X_state[si+oi] + dt*(dX_state[si+oi] - term) < 0:
                            term = X_state[si+oi]*r_dt
                            dX_state[to_node.state_index+oi] += term + dX_state[si+oi] # We shift the SIR dynamics that transpired in the cnode into the node
                            dX_state[si+oi] += - (term + dX_state[si+oi])
                        # Otherwise apply the transport as usual
                        else:
                            dX_state[si+oi] -= term
                            dX_state[to_node.state_index+oi] += term
            else:
                cTs[cTi].is_on = False
                cn = cnodes[cTs[cTi].cnode_index]
                cn.is_on = False # Turn off commuter node
                

        ################################################################
        #### Forward-Euler #############################################
        ################################################################
        
        for j in prange(X_state_size, nogil=True):
            X_state[j] += dX_state[j]*dt

        t += dt
        
        if steps_per_print != -1 and step_i % steps_per_print==0:
            print("Step %s out of %s" % (step_i, steps))
        
        #### Store state

        if steps_per_save != -1:

            if (step_i+1) % steps_per_save == 0:
                X_states_saved[save_i,:] = X_state[:X_states_saved_col_num]
                ts_saved[save_i] = t
                save_i += 1

        #### Call event function

        if is_a_event_on[step_i]:
            event_i = 0
            for event_function in event_functions:
                if is_event_on[step_i][event_i]:
                    if steps_per_save == -1:
                        event_function(self, step_i, t, dt, X_state, dX_state)
                    else:
                        event_function(self, step_i, t, dt, X_state, dX_state, X_states_saved, ts_saved, save_i)
                event_i += 1

        #### Call Cython event function

        #if cevent_steps[step_i]:
        #    cevent_function(self, step_i, t, dt, X_state, dX_state)

    free(loc_j_is_stochastic)
    free(to_k_is_stochastic)

    if steps_per_save != -1:

        node_mappings = self.node_mappings.copy()
        cnode_mappings = self.cnode_mappings.copy()

        sim_data = (node_mappings, cnode_mappings, ts_saved, X_states_saved)

        if save_path != '':
            with open("%s/%s" % (save_path, save_node_mappings_path),"wb") as f:
                pickle.dump(node_mappings, f)
            with open("%s/%s" % (save_path, save_cnode_mappings_path),"wb") as f:
                pickle.dump(cnode_mappings, f)
            np.save("%s/%s" % (save_path, save_ts_path), ts_saved)

        return sim_data

cdef contact_scaling_linear(DTYPE_t rho, DTYPE_t[:] params):
    """a + b*rho"""
    return params[0] + params[1]*rho

cdef contact_scaling_powerlaw(DTYPE_t rho, DTYPE_t[:] params):
    """a + b*rho^c"""
    return params[0] + params[1] * libc.math.pow(rho, params[2])

cdef contact_scaling_exp(DTYPE_t rho, DTYPE_t[:] params):
    """a * b^rho"""
    return params[0] * libc.math.pow(params[1], rho)

cdef contact_scaling_log(DTYPE_t rho, DTYPE_t[:] params):
    """log(a + b*rho)"""
    return libc.math.log(params[0] + params[1]*rho)