import zarr
import pickle
import cython
from libc.math cimport exp
from libc.stdlib cimport malloc, free
from cython.parallel import prange
import numpy as np
cimport numpy as np
import time # For seeding random
import scipy.stats

from pyrossgeo.__defs__ import DTYPE

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef simulate(Simulation self, DTYPE_t[:] X_state, DTYPE_t t_start, DTYPE_t t_end, object _dts, int steps_per_save=-1,
                            str save_path="", bint only_save_nodes=False, int steps_per_print=-1,
                            object event_times=[], object event_function=None,
                            int random_seed=-1):
                            #object cevent_times=[], SIM_EVENT cevent_function=SIM_EVENT_NULL):
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

    # Model
    cdef model_term* model_linear_terms = self.model_linear_terms
    cdef int model_linear_terms_len = self.model_linear_terms_len
    cdef model_term* model_infection_terms = self.model_infection_terms
    cdef int model_infection_terms_len = self.model_infection_terms_len
    cdef int[:] infection_classes_indices = self.infection_classes_indices
    cdef int infection_classes_num = self.infection_classes_num
    cdef DTYPE_t[:,:,:] contact_matrices = self.contact_matrices
    cdef int contact_matrices_num = self.contact_matrices.shape[0]

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
    total_Os_arr = np.zeros(model_dim, dtype=DTYPE)
    cdef DTYPE_t[:] total_Os = total_Os_arr # Used to see whether stochasticity should be turned on
<<<<<<< HEAD
    #cdef DTYPE_t[:] stochastic_threshold_from_below = self.stochastic_threshold_from_below
    #cdef DTYPE_t[:] stochastic_threshold_from_above = self.stochastic_threshold_from_abov
    cdef bint* loc_j_is_stochastic
    cdef bint* to_k_is_stochastic

    stochastic_threshold_from_below_arr = np.array( [10000000, 10000000, 10000000, 10000000, 10000000] )
    stochastic_threshold_from_above_arr = np.array( [500, 500, 500, 500, 500] )
    stochastic_threshold_from_below = stochastic_threshold_from_below_arr
    stochastic_threshold_from_above = stochastic_threshold_from_above_arr
=======
    #cdef DTYPE_t[:] stoch_threshold_from_below = self.stoch_threshold_from_below
    #cdef DTYPE_t[:] stoch_threshold_from_above = self.stoch_threshold_from_abov
    cdef bint* loc_j_is_stochastic
    cdef bint* to_k_is_stochastic

    stoch_threshold_from_below_arr = np.array( [100, 100, 100, 100, 100] )
    stoch_threshold_from_above_arr = np.array( [50, 50, 50, 50, 50] )
    stoch_threshold_from_below = stoch_threshold_from_below_arr
    stoch_threshold_from_above = stoch_threshold_from_above_arr
>>>>>>> Implemented stochastic protocol
    
    if random_seed == -1:
        random_seed = np.int64(np.round(time.time()))
    cdef mt19937 gen = mt19937(random_seed)
    cdef poisson_distribution[int] dist

    # Consts
    cdef int minutes_in_day = 1440
    save_node_mappings_path = 'node_mappings.pkl'
    save_cnode_mappings_path = 'cnode_mappings.pkl'
    save_ts_path = 'ts.npy'
    save_X_states_path = 'X_states.zarr'

    #### Simulation variables ##########################################

    cdef int cni, ni, si, Ti, cTi, i, j, ui, u, o, cmat_i, oi, X_index, loc_j, to_k, age_a, age_b
    cdef bint to_k_is_active = False
    cdef DTYPE_t S, t1, t2, transport_profile, fro_N, cn_N, term, transport_profile_exponent
    cdef node n, fro_n, to_n
    cdef cnode cn
    cdef model_term mt
    
    cdef DTYPE_t[:, :, :] _lambdas = self._lambdas
    cdef DTYPE_t[:] _Is = self._Is
    cdef DTYPE_t[:] _Ns = self._Ns

    cdef np.ndarray dX_state_arr
    cdef DTYPE_t[:] dX_state

    #### Initialize stochasticity ######################################

    loc_j_is_stochastic = <bint *> malloc( (self.max_node_index+1) * sizeof(bint))
    to_k_is_stochastic = <bint *> malloc( (self.max_node_index+1) * sizeof(bint))

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
<<<<<<< HEAD
            loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stochastic_threshold_from_below[o])
=======
            loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stoch_threshold_from_below[o])
>>>>>>> Implemented stochastic protocol

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
<<<<<<< HEAD
            to_k_is_stochastic[to_k] = to_k_is_stochastic[to_k] or (total_Os[o] < stochastic_threshold_from_below[o])
=======
            to_k_is_stochastic[to_k] = to_k_is_stochastic[to_k] or (total_Os[o] < stoch_threshold_from_below[o])
>>>>>>> Implemented stochastic protocol

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

    cdef np.ndarray event_steps_arr = np.full(steps, 0, dtype=np.int8)
    cdef char[:] event_steps = event_steps_arr

    #cdef np.ndarray cevent_steps_arr = np.full(steps, 0, dtype=np.int8)
    #cdef char[:] cevent_steps = cevent_steps_arr

    t = t_start
    for step_i in range(steps):
        dt = dts[step_i % dts_num]
        t += dt

        for et in event_times:
            if et <= t and t < et+dt:
                event_steps[step_i] = 1

        #for et in cevent_times:
        #    if et <= t and t < et+dt:
        #        cevent_steps[step_i] = 1

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

        #### Node dynamics #############################################

        for loc_j in range(max_node_index+1):

            # Compute the total populations of each class at loc_j, as well as the populaitons of each age group

            for o in range(model_dim):
                total_Os[o] = 0

            for age_a in range(age_groups):
                _Ns[age_a] = 0
                for i in range(nodes_at_j_len[age_a][loc_j]):
                    n = nodes[nodes_at_j[age_a][loc_j][i]]
                    for o in range(model_dim):
                        total_Os[o] += X_state[n.state_index + o]
                        _Ns[age_a] += X_state[n.state_index + o]

            #### Compute lambdas

            for ui in range(infection_classes_num):
                u = infection_classes_indices[ui]

                # Find the infecteds of each age group

                for age_a in range(age_groups):
                    _Is[age_a] = 0
                    for i in range(nodes_at_j_len[age_a][loc_j]):
                        n = nodes[nodes_at_j[age_a][loc_j][i]]
                        _Is[age_a] += X_state[n.state_index + u]

                # Compute lambdas

                for cmat_i in range(contact_matrices_num):
                    for age_a in range(age_groups):
                        _lambdas[cmat_i][age_a][ui] = 0
                        for age_b in range(age_groups):
                            if _Ns[age_b] > 1: # No infections can occur if there are fewer than one person at node
                                _lambdas[cmat_i][age_a][ui] += contact_matrices[cmat_i][age_a][age_b] * _Is[age_b] / _Ns[age_b]

            #### Decide whether deterministic or stochastic

            if loc_j_is_stochastic[loc_j]:
                loc_j_is_stochastic[loc_j] = False
                for o in range(model_dim):
<<<<<<< HEAD
                    loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stochastic_threshold_from_below[o])
            else:
                loc_j_is_stochastic[loc_j] = False
                for o in range(model_dim):
                    loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stochastic_threshold_from_above[o])
=======
                    loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stoch_threshold_from_below[o])
            else:
                loc_j_is_stochastic[loc_j] = False
                for o in range(model_dim):
                    loc_j_is_stochastic[loc_j] = loc_j_is_stochastic[loc_j] or (total_Os[o] < stoch_threshold_from_above[o])
>>>>>>> Implemented stochastic protocol

            #### Compute the derivatives at each node

            # Stochastic
            if True and loc_j_is_stochastic[loc_j]:
                for age_a in range(age_groups):
                    for i in range(nodes_at_j_len[age_a][loc_j]): 
                        n = nodes[nodes_at_j[age_a][loc_j][i]]
                        si = n.state_index
                        S = X_state[si] # S is always located at the state index

                        for j in range(model_linear_terms_len):
                            mt = model_linear_terms[j]   
                            if X_state[si+mt.oi_coupling] > 0: # Only allow interaction if the class is positive
<<<<<<< HEAD
                                #dist = poisson_distribution[int](dt*n.linear_coeffs[j]*X_state[si+mt.oi_coupling])
                                #term = dist(gen) * r_dt
                                term = scipy.stats.poisson.rvs(dt*n.linear_coeffs[j]*X_state[si+mt.oi_coupling]) * r_dt
=======
                                dist = poisson_distribution[int](dt*n.linear_coeffs[j]*X_state[si+mt.oi_coupling])
                                term = dist(gen) * r_dt
                                #term = scipy.stats.poisson.rvs(dt*n.linear_coeffs[j]*X_state[si+mt.oi_coupling]) * r_dt
>>>>>>> Implemented stochastic protocol
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

                        for j in range(model_infection_terms_len):
                            mt = model_infection_terms[j]
                            if _lambdas[cmat_i][age_a][mt.infection_index] > 0: # Only allow interaction if the class is positive
                                cmat_i = n.contact_matrix_indices[mt.infection_index]
<<<<<<< HEAD
                                #dist = poisson_distribution[int](dt*n.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S)
                                #term = dist(gen) * r_dt
                                term = scipy.stats.poisson.rvs(dt*n.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S) * r_dt
=======
                                dist = poisson_distribution[int](dt*n.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S)
                                term = dist(gen) * r_dt
                                #term = scipy.stats.poisson.rvs(dt*n.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S) * r_dt
>>>>>>> Implemented stochastic protocol
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term            
            # Deterministic
            else:
                for age_a in range(age_groups):
                    for i in range(nodes_at_j_len[age_a][loc_j]): 
                        n = nodes[nodes_at_j[age_a][loc_j][i]]
                        si = n.state_index
                        S = X_state[si] # S is always located at the state index

                        for j in range(model_linear_terms_len):
                            mt = model_linear_terms[j]
                            if X_state[si+mt.oi_coupling] > 0: # Only allow interaction if the class is positive
                                term = n.linear_coeffs[j] * X_state[si+mt.oi_coupling]
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

                        for j in range(model_infection_terms_len):
                            mt = model_infection_terms[j]
<<<<<<< HEAD
                            if dX_state[si+mt.oi_neg] > 0:
                            #if _lambdas[cmat_i][age_a][mt.infection_index] > 0: # Only allow interaction if the class is positive
=======
                            if _lambdas[cmat_i][age_a][mt.infection_index] > 0: # Only allow interaction if the class is positive
>>>>>>> Implemented stochastic protocol
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

            # Compute the total populations of each class at loc_j, as well as the populaitons of each age group

            for o in range(model_dim):
                total_Os[o] = 0

            for age_a in range(age_groups):
                _Ns[age_a] = 0
                for i in range(cnodes_into_k_len[age_a][to_k]):
                    cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                    for o in range(model_dim):
                        total_Os[o] += X_state[cn.state_index + o]
                        _Ns[age_a] += X_state[cn.state_index + o]

            #### Compute lambdas

            for ui in range(infection_classes_num):
                u = infection_classes_indices[ui]

                # Find the infecteds of each age group

                for age_a in range(age_groups):
                    _Is[age_a] = 0
                    for i in range(cnodes_into_k_len[age_a][to_k]):
                        cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                        _Is[age_a] += X_state[cn.state_index + u]

                # Compute lambdas

                for cmat_i in range(contact_matrices_num):
                    for age_a in range(age_groups):
                        _lambdas[cmat_i][age_a][ui] = 0
                        for age_b in range(age_groups):
                            if _Ns[age_b] > 1: # No infections can occur if there are fewer than one person at node
                                _lambdas[cmat_i][age_a][ui] += contact_matrices[cmat_i][age_a][age_b] * _Is[age_b] / _Ns[age_b]

            #### Decide whether deterministic or stochastic

            if to_k_is_stochastic[loc_j]:
                to_k_is_stochastic[loc_j] = False
                for o in range(model_dim):
<<<<<<< HEAD
                    to_k_is_stochastic[loc_j] = to_k_is_stochastic[loc_j] or (total_Os[o] < stochastic_threshold_from_below[o])
            else:
                to_k_is_stochastic[loc_j] = True
                for o in range(model_dim):
                    to_k_is_stochastic[loc_j] = to_k_is_stochastic[loc_j] and (total_Os[o] > stochastic_threshold_from_above[o])

            # Stochastic
            if True and loc_j_is_stochastic[to_k]:
=======
                    to_k_is_stochastic[loc_j] = to_k_is_stochastic[loc_j] or (total_Os[o] < stoch_threshold_from_below[o])
            else:
                to_k_is_stochastic[loc_j] = True
                for o in range(model_dim):
                    to_k_is_stochastic[loc_j] = to_k_is_stochastic[loc_j] and (total_Os[o] > stoch_threshold_from_above[o])

            # Stochastic
            if False and loc_j_is_stochastic[to_k]:
>>>>>>> Implemented stochastic protocol
                for age_a in range(age_groups):
                    for i in range(cnodes_into_k_len[age_a][to_k]): 
                        cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                        si = cn.state_index
                        S = X_state[si] # S is always located at the state index

                        for j in range(model_linear_terms_len):
                            mt = model_linear_terms[j]
                            if X_state[si+mt.oi_coupling] > 0: # Only allow interaction if the class is positive
<<<<<<< HEAD
                                #dist = poisson_distribution[int](dt*cn.linear_coeffs[j]*X_state[si+mt.oi_coupling])
                                #term = dist(gen) * r_dt
                                term = scipy.stats.poisson.rvs(dt*cn.linear_coeffs[j]*X_state[si+mt.oi_coupling]) * r_dt
=======
                                dist = poisson_distribution[int](dt*cn.linear_coeffs[j]*X_state[si+mt.oi_coupling])
                                term = dist(gen) * r_dt
                                #term = scipy.stats.poisson.rvs(dt*cn.linear_coeffs[j]*X_state[si+mt.oi_coupling]) * r_dt
>>>>>>> Implemented stochastic protocol
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

                        for j in range(model_infection_terms_len):
                            mt = model_infection_terms[j]
                            if _lambdas[cmat_i][age_a][mt.infection_index] > 0: # Only allow interaction if the class is positive
                                cmat_i = cn.contact_matrix_indices[mt.infection_index]
<<<<<<< HEAD
                                #dist = poisson_distribution[int](dt*cn.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S)
                                #term = dist(gen) * r_dt
                                term = scipy.stats.poisson.rvs(dt*cn.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S) * r_dt
=======
                                dist = poisson_distribution[int](dt*cn.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S)
                                term = dist(gen) * r_dt
                                #term = scipy.stats.poisson.rvs(dt*cn.infection_coeffs[j]*_lambdas[cmat_i][age_a][mt.infection_index]*S) * r_dt
>>>>>>> Implemented stochastic protocol
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term
            # Deterministic
            else:
<<<<<<< HEAD
                print(123123)
=======
>>>>>>> Implemented stochastic protocol
                for age_a in range(age_groups):
                    for i in range(cnodes_into_k_len[age_a][to_k]): 
                        cn = cnodes[cnodes_into_k[age_a][to_k][i]]
                        si = cn.state_index
                        S = X_state[si] # S is always located at the state index

                        for j in range(model_linear_terms_len):
                            mt = model_linear_terms[j]
                            if X_state[si+mt.oi_coupling] > 0: # Only allow interaction if the class is positive
                                term = cn.linear_coeffs[j] * X_state[si+mt.oi_coupling]
                                dX_state[si+mt.oi_pos] += term
                                dX_state[si+mt.oi_neg] -= term

                        for j in range(model_infection_terms_len):
                            mt = model_infection_terms[j]
                            if _lambdas[cmat_i][age_a][mt.infection_index] > 0: # Only allow interaction if the class is positive
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
                transport_profile = exp(- transport_profile_exponent * transport_profile_exponent * transport_profile_c_r) * transport_profile_integrated_r * Ts[Ti].r_T_Delta_t
                
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
                transport_profile = exp(- transport_profile_exponent * transport_profile_exponent * transport_profile_c_r) * transport_profile_integrated_r * cTs[cTi].r_T_Delta_t

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

        if event_steps[step_i]:
            if steps_per_save == -1:
                event_function(self, step_i, t, dt, X_state, dX_state)
            else:
                event_function(self, step_i, t, dt, X_state, dX_state, X_states_saved, ts_saved, save_i)

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
            with open("%s/%s" % (save_path, save_node_mappings_path),"wb") as f: pickle.dump(node_mappings, f)
            with open("%s/%s" % (save_path, save_cnode_mappings_path),"wb") as f: pickle.dump(cnode_mappings, f)
            np.save("%s/%s" % (save_path, save_ts_path), ts_saved)

        return sim_data
