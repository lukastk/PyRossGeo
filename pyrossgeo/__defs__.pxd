import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_t

cdef struct node:
    #Contains relevant information for each node.
    int home
    int loc
    int age
    int state_index # The index of the node's value in X_state
    int* incoming_T_indices
    int incoming_T_indices_len
    int* outgoing_T_indices
    int outgoing_T_indices_len
    int* contact_matrix_indices
    DTYPE_t** linear_coeffs
    DTYPE_t** infection_coeffs

cdef struct cnode:
    #Contains relevant information for each commuter node.
    int home
    int fro
    int to
    int age
    int state_index # The index of the node's value in X_state
    int incoming_node # Index of incoming node
    int outgoing_node # Index of outgoing node
    int incoming_T
    int outgoing_T
    int* contact_matrix_indices
    DTYPE_t area
    bint is_on
    DTYPE_t** linear_coeffs
    DTYPE_t** infection_coeffs

cdef struct transporter:
    #Represents a commuting schedule.
    int T_index
    int age
    int home
    int fro
    int to
    int fro_node_index # The index of the origin node
    int to_node_index # The index of the destination node
    int cnode_index # The index of the commuting node
    DTYPE_t t1
    DTYPE_t t2
    DTYPE_t r_T_Delta_t
    DTYPE_t move_N
    DTYPE_t move_percentage
    bint use_percentage
    bint* moving_classes
    bint is_on # When true, the transport is on
    DTYPE_t N0