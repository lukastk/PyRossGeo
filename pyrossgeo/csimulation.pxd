import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_t

ctypedef void (*SIM_EVENT)(csimulation cg, int step_i, DTYPE_t t, DTYPE_t dt, DTYPE_t[:] X_state, DTYPE_t[:] dX_state)
cdef void SIM_EVENT_NULL(csimulation cg, int step_i, DTYPE_t t, DTYPE_t dt, DTYPE_t[:] X_state, DTYPE_t[:] dX_state)

cdef class csimulation:

    cdef readonly int age_groups
    cdef readonly int model_dim
    cdef readonly int max_node_index

    cdef node* nodes
    cdef int nodes_num

    cdef cnode* cnodes
    cdef int cnodes_num

    cdef int state_size
    cdef int node_states_len # The slice of X_states which only comprises nodes

    cdef readonly object state_mappings

    cdef readonly dict storage # Persistent storage that will be used for events

    # Model
    cdef int** class_infections
    cdef int* class_infections_num
    cdef int[:] infection_classes_indices
    cdef int infection_classes_num
    cdef int** linear_terms
    cdef int* linear_terms_num
    cdef np.ndarray contact_matrices
    cdef readonly dict contact_matrices_key_to_index
    cdef np.ndarray node_infection_cmats
    cdef np.ndarray cnode_infection_cmats
    cdef object _lambdas_arr
    cdef np.ndarray _lambdas
    cdef object _Is_arr
    cdef np.ndarray _Is
    cdef object _Ns_arr
    cdef np.ndarray _Ns

    # Transport
    cdef transporter* Ts # Going into commuterverses
    cdef int Ts_num
    cdef transporter* cTs # Going out from commuterverses
    cdef int cTs_num

    # Used for lambda calculation
    cdef int*** nodes_at_j
    cdef int** nodes_at_j_len

    # Used for tau calculation
    cdef int*** cnodes_into_k
    cdef int** cnodes_into_k_len

    # Transport profile
    cdef DTYPE_t transport_profile_integrated
    cdef DTYPE_t transport_profile_integrated_r
    cdef DTYPE_t transport_profile_m
    cdef DTYPE_t transport_profile_c
    cdef DTYPE_t transport_profile_c_r

    cpdef get_contact_matrix_keys(self)
    cpdef get_contact_matrix(self, str cmat_key)
    cpdef set_contact_matrix(self, str cmat_key, np.ndarray cmat)

    cpdef stop_commuting(self, bint s)
    cpdef bint is_commuting_stopped(self)

cdef struct node:
    int home
    int loc
    int age
    int state_index
    int* incoming_T_indices
    int incoming_T_indices_len
    int* outgoing_T_indices
    int outgoing_T_indices_len
    DTYPE_t** linear_coeffs
    DTYPE_t** infection_coeffs

cdef struct cnode:
    int home
    int fro
    int to
    int age
    int state_index
    int incoming_node
    int outgoing_node
    int incoming_T
    int outgoing_T
    DTYPE_t** linear_coeffs
    DTYPE_t** infection_coeffs

cdef struct transporter:
    int T_index
    int age
    int home
    int fro
    int to
    int fro_node_index
    int to_node_index
    int cnode_index
    DTYPE_t t1
    DTYPE_t t2
    DTYPE_t r_T_Delta_t
    DTYPE_t move_N
    DTYPE_t move_percentage
    bint use_percentage
    bint* moving_classes
    bint is_on # When true, the transport is on
    DTYPE_t N0