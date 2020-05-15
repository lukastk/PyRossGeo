import numpy as np
import pickle
import zarr

def extract_node_data(sim_data):
    """Returns the results of the simulation for each node.

    Parameters
    ----------
        sim_data : Tuple
            Simulation data.

    Returns
    -------
    dict
        A dictionary of keys of the form `(i, j)`, corresponding to
        home node, and location node respectively.
        `node_data[i,j,k]` is an `np.ndarray` of shape
        `(ts.size, # of age groups, # of classes)`.
    """
    node_mappings, cnode_mappings, ts, X_states = sim_data
    node_data = {}

    age_groups = 0
    model_dim = 0
    for a,o,i,j in node_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
    for a,o,i,j,k in cnode_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1

    for a,o,i,j in node_mappings:
        index = node_mappings[a,o,i,j]

        if not (i,j) in node_data:
            node_data[i,j] = np.zeros( (len(ts), age_groups, model_dim) )

        node_data[i,j][:,a,o] = X_states[:,index]

    return node_data

def extract_cnode_data(sim_data):
    """Returns the results of the simulation for each commuter node.

    Parameters
    ----------
        sim_data : Tuple
            Simulation data.

    Returns
    -------
    dict
        A dictionary of keys of the form `(i, j, k)`, corresponding to
        home node, origin node and destination node respectively.
        `cnode_data[i,j,k]` is an `np.ndarray` of shape
        `(ts.size, # of age groups, # of classes)`.
    """
    node_mappings, cnode_mappings, ts, X_states = sim_data
    cnode_data = {}

    age_groups = 0
    model_dim = 0
    for a,o,i,j in node_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
    for a,o,i,j,k in cnode_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1

    for a,o,i,j,k in cnode_mappings:
        index = cnode_mappings[a,o,i,j,k]

        if not (i,j,k) in cnode_data:
            cnode_data[i,j,k] = np.zeros( (len(ts), age_groups, model_dim) )

        cnode_data[i,j,k][:,a,o] = X_states[:,index]

    return cnode_data

def extract_network_data(sim_data):
    """Returns the results of the simulation for the whole network.

    Parameters
    ----------
        sim_data : Tuple
            Simulation data.

    Returns
    -------
    np.ndarray
        An array of shape (ts.size, # of age groups, # of classes).
        It contains the result of the simulation of the network as a whole
        for each age group and class.
    """
    node_mappings, cnode_mappings, ts, X_states = sim_data

    age_groups = 0
    model_dim = 0
    for a,o,i,j in node_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
    for a,o,i,j,k in cnode_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1

    network_data = np.zeros( (len(ts), age_groups, model_dim) )

    for a,o,i,j in node_mappings:
        network_data[:,a,o] += X_states[:,node_mappings[a,o,i,j]]

    for a,o,i,j,k in cnode_mappings:
        network_data[:,a,o] += X_states[:,cnode_mappings[a,o,i,j,k]]

    return network_data

def extract_location_data(sim_data):
    """Returns the results of the simulation for a given location.

    Parameters
    ----------
        sim_data : Tuple
            Simulation data.

    Returns
    -------
    np.ndarray
        An array of shape (ts.size, # of age groups, # of classes,
        # of locations). It contains the results of the simulation at each
        location. So `community_data[5,0,1,32]` contains the state of
        people of age-bracket 0, class 1 who are at location 32, at step 5
        of the simulation.
    """
    node_mappings, cnode_mappings, ts, X_states = sim_data

    age_groups = 0
    model_dim = 0
    max_loc_index = 0

    for a,o,i,j in node_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
        if j > max_loc_index:
            max_loc_index = j
    for a,o,i,j,_ in cnode_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
        if j > max_loc_index:
            max_loc_index = j

    location_data = np.zeros( (len(ts), age_groups, model_dim, max_loc_index+1) )

    for a,o,i,j in node_mappings:
        location_data[:,a,o,j] += X_states[:,node_mappings[a,o,i,j]]

    return location_data

def extract_community_data(sim_data):
    """Returns the results of the simulation for each community.

    Parameters
    ----------
        sim_data : Tuple
            Simulation data.

    Returns
    -------
    np.ndarray
        An array of shape (ts.size, # of age groups, # of classes,
        # of locations). It contains the results of the simulation summed
        over each community. So `community_data[:,0,1,32]` contains the
        history of all people of age-bracket 0, class 1 and who live at location 32.
    """
    node_mappings, cnode_mappings, ts, X_states = sim_data

    age_groups = 0
    model_dim = 0
    max_home_index = 0

    for a,o,i,j in node_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
        if i > max_home_index:
            max_home_index = i
    for a,o,i,j,k in cnode_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
        if i > max_home_index:
            max_home_index = i

    community_data = np.zeros( (len(ts), age_groups, model_dim, max_home_index+1) )

    for a,o,i,j in node_mappings:
        community_data[:,a,o,i] += X_states[:,node_mappings[a,o,i,j]]

    for a,o,i,j,k in cnode_mappings:
        community_data[:,a,o,i] += X_states[:,cnode_mappings[a,o,i,j,k]]

    return community_data

def extract_simulation_data(sim_data):
    """Returns a tuple containing various formatted data for a given simulation result.

    It returns `node_data, cnode_data, location_data, community_data, network_data`.
    """
    node_mappings, cnode_mappings, ts, _ = sim_data

    node_data = extract_node_data(sim_data)
    cnode_data = extract_cnode_data(sim_data)

    age_groups = 0
    model_dim = 0
    max_home_index = 0
    max_loc_index = 0

    for a,o,i,j in node_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
        if i > max_home_index:
            max_home_index = i
        if j > max_loc_index:
            max_loc_index = j
    for a,o,i,j,k in cnode_mappings:
        if a+1 > age_groups:
            age_groups = a+1
        if o+1 > model_dim:
            model_dim = o+1
        if i > max_home_index:
            max_home_index = i
        if j > max_loc_index:
            max_loc_index = j

    location_data = np.zeros( (len(ts), age_groups, model_dim, max_loc_index+1) )
    community_data = np.zeros( (len(ts), age_groups, model_dim, max_home_index+1) )
    network_data = np.zeros( (len(ts), age_groups, model_dim) )

    for i,j in node_data:
        node_data_ij = node_data[i,j]
        location_data[:, :, :, j] += node_data_ij
        community_data[:, :, :, i] += node_data_ij
        network_data[:, :, :] += node_data_ij

    for i,j,k in cnode_data:
        cnode_data_ijk = cnode_data[i,j,k]
        community_data[:, :, :, i] += cnode_data_ijk
        network_data[:, :, :] += cnode_data_ijk

    return ts, node_data, cnode_data, location_data, community_data, network_data

def extract_ts(sim_data):
    """Returns the results of the simulation times given simulation data.

    Parameters
    ----------
        sim_data : Tuple
            Simulation data.

    Returns
    -------
    np.ndarray
        A 1D array containing each time-step.
    """
    ts_saved = sim_data[2]
    return ts_saved

def load_sim_data(load_path, use_zarr=False):
    """Loads 

    Parameters
    ----------
        load_path : str
            Path of the simulation data folder.
        use_zarr : bool, optional
            If True, the simulation data will be given as a zarr array,
            rather than as a numpy array. The former is useful if the
            data is very large.

    Returns
    -------
    tuple
        A tuple `(node_mappings, cnode_mappings, ts, X_states)`, containing
        all simulation data. `X_states` is either an `np.ndarray` or a `zarr.core.Array`.
        If `use_zarr=True`, the latter will be given.
    """
    node_mappings_path = 'node_mappings.pkl'
    cnode_mappings_path = 'cnode_mappings.pkl'
    ts_path = 'ts.npy'
    X_states_path = 'X_states.zarr'

    node_mappings = pickle.load( open( "%s/%s" %( load_path, node_mappings_path ), "rb" ) )
    cnode_mappings = pickle.load( open( "%s/%s" %( load_path, cnode_mappings_path ), "rb" ) )
    ts = np.load("%s/%s" %( load_path, ts_path ))
    X_states = zarr.open( "%s/%s" %( load_path, X_states_path ) , chunks=(len(ts), 1))

    if not use_zarr:
        X_states = X_states[:]

    sim_data = ( node_mappings, cnode_mappings, ts, X_states )
    return sim_data

def get_dt_schedule(times, end_time):
    """Generates a time-step schedule.

    Example:

    The following generates a time-step schedule where we use a time-step
    of one minute between 7-10 and 17-19 o\'clock, and 2 hours for all
    other times.

        ts, dts = pyrossgeo.utils.get_dt_schedule([
            (0,  2*60),
            (7*60,  1),
            (10*60, 2*60),
            (17*60, 1),
            (19*60, 2*60)
            ], end_time=24*60)

    Parameters
    ----------
        times : lost
            list of tuples
        end_time : float
            The final time of the schedule.

    Returns
    -------
        tuple
            A tuple `(ts, dts)`. `dts` are the time-steps and `ts`
            the times.
    """
    times = list(times)
    times.append( (end_time, 0) )
    ts = []

    for i in range(len(times)-1):
        t, dt = times[i]
        t_next = times[i+1][0]
        ts.append(np.arange(t, t_next, dt))
        
    ts.append([end_time])
    ts = np.concatenate(ts)
    dts = (ts - np.roll(ts, 1))[1:]
        
    return np.array(ts, dtype=np.double), np.array(dts, dtype=np.double)