import numpy as np
cimport numpy as np

from pyrossgeo._initialization import initialize
from pyrossgeo._simulation cimport simulate
from pyrossgeo._helpers import compute, free_sim

#cdef void SIM_EVENT_NULL(Simulation sim, int step_i, DTYPE_t t, DTYPE_t dt, DTYPE_t[:] X_state, DTYPE_t[:] dX_state):
#    raise(Exception('SIM_EVENT_NULL triggered'))

cdef class Simulation:

    def __cinit__(self):
        self.storage = {}
        self.has_been_initialized = False

    def __dealloc__(self):
        free_sim(self)

    def initialize(self, sim_config_path='', model_dat='', commuter_networks_dat='',
                        node_parameters_dat='', cnode_parameters_dat='',
                        contact_matrices_dat='', node_cmatrices_dat='',
                        cnode_cmatrices_dat='', node_populations_dat='',
                        cnode_populations_dat=''):
        """Initializes the simulation using the given configuration files.
        
        Each argument will be accepted as either as raw data or a path
        to a file containing the data. See the documentation for the
        format each configuration file should take.

        Args:
            model_dat : str or dict
                Specifies the epidemic model
            commuter_networks_dat : str or np.ndarray
                Specifies the commuter network
            node_parameters_dat : str or pandas.DataFrame
                Specifies the parameters of the model at each node
            cnode_parameters_dat : str or pandas.DataFrame
                Specifies the parameters of the model at each commuter node
            contact_matrices_dat : str or dict
                Specifies the contact matrices
            node_cmatrices_dat : str or pandas.DataFrame
                Specifies what contact matrix to use at each node
            cnode_cmatrices_dat : str or pandas.DataFrame
                Specifies what contact matrix to use at each commuter node
            node_populations_dat : str or np.ndarray
                Specifies the population at each node
            cnode_populations_dat : str or np.ndarray
                Specifies the population at each commuter node (default None)
        """
        return initialize(self, sim_config_path, model_dat, commuter_networks_dat,
                        node_parameters_dat, cnode_parameters_dat,
                        contact_matrices_dat, node_cmatrices_dat,
                        cnode_cmatrices_dat, node_populations_dat,
                        cnode_populations_dat)

    def simulate(self, X_state, t_start, t_end, dts, steps_per_save=-1,
                    save_path="", only_save_nodes=False, steps_per_print=-1,
                    event_times=[], event_function=None):
        """Simulates the system.
        
        Simulates the system between times `t_start` and `t_end`, with the 
        initial condition `X_state`. `dts` an array of time-steps, where
        `dts[i]` is the time-step used during step `i` of the simualtion.

        If `save_path` is specified, then the result of the simulation 
        will be outputted directly to the hard-drive as a .zarr array
        at the given path.

        Parameters
        ----------
            X_state : np.ndarray
                Initial condition of the system.
            t_start : float
                Start time
            t_end : float
                End time
            dts : float, or list or array of floats
                Time steps
            steps_per_save : int
                Number of simulation steps per saving the state (default -1)
            save_path : str
                The path of the folder to save the output to (default "")
            only_save_nodes : bool
                If True, commuter nodes will not be saved (default False)
            event_times : list or array of floats
                The times at which the `event_function` should be called (default [])
            event_function : function
                The function that will be called at each event time (default None)

        Returns
        -------
            A tuple `((node_mappings, cnode_mappings), ts, X_states)`. `X_states` is an
            np.ndarray of shape `(ts.size, N)` where `N` is the total
            degrees of freedom of the system. If `only_save_nodes = True`
            then `N` is just the degrees of freedom of the nodes, and excludes
            the commuting nodes. `ts` is the times corresponding to `X_states`.
            `node_mappings` is a dictionary with keys of form `(a,o,i,j)`,
            corresponding to age-bracket, class, home and location respectively.
            `node_mappings[a,o,i,j]` is the column of `X_states` for the 
            corresponding state value. Similarly, `cnode_mappings` is
            a dictionary with keys of the form `(a,o,i,j,k)`, corresponding
            to age-bracket, class, home, origin, destination respectively.
        """
        return simulate(self, X_state, t_start, t_end, dts, steps_per_save,
                    save_path, only_save_nodes, steps_per_print,
                    event_times, event_function)

    def compute(self, X_state, dX_state, t, dt):
        """Computes the right-hand side of the dynamical system.
        
        Sets the array `dX_state` to the derivative of the dynamical system
        at the state `X_state`.

        Parameters
        ----------
            X_state : np.ndarray
                The state of the system.
            dX_state : np.ndarray
                The array to input the derivative into.
            t : float
                Time
            dt : float
                The time-step used
        """
        compute(self, X_state, dX_state, t, dt)

    cpdef get_contact_matrix_keys(self):
        """Returns a list of the contact matrix keys."""
        return list(self.contact_matrices_key_to_index.keys())

    cpdef get_contact_matrix(self, str cmat_key):
        """Returns the contact matrix with the given key."""
        return self.contact_matrices[self.contact_matrices_key_to_index[cmat_key]]

    cpdef set_contact_matrix(self, str cmat_key, np.ndarray cmat):
        """Change the contact matrix of the given key.
        
        Args:
            cmat_key: the key of the contact matrix
            cmat: the array to set the contact matrix to.
        """
        self.contact_matrices[self.contact_matrices_key_to_index[cmat_key]] = cmat

    cpdef stop_commuting(self, bint s):
        """Disables the commuter network if s is True, and enables it if False."""

        # Turn off commuting
        if not self.is_commuting_stopped() and s:
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
        # Turn on commuting
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
        """Returns True if the commuter network has been disabled."""
        if 'commuting_is_stopped' in self.storage:
            return self.storage['commuting_is_stopped']
        else:
            return False

    #TODO
    # - function that returns what contact matrix is sued for a specific node
    # - see what the parameters are at a specific node (in dictionary format), and edit them
    # - 