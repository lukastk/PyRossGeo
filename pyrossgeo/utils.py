import numpy as np

def get_simulation_data(sim_data):
    state_mappings, ts, _ = sim_data
    node_mappings, cnode_mappings = state_mappings

    node_data = get_node_data(sim_data)
    cnode_data = get_cnode_data(sim_data)

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

def get_node_data(sim_data):
    state_mappings, ts, X_states = sim_data
    node_mappings, cnode_mappings = state_mappings
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

def get_cnode_data(sim_data):
    state_mappings, ts, X_states = sim_data
    node_mappings, cnode_mappings = state_mappings
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

        try:
            cnode_data[i,j,k][:,a,o] = X_states[:,index]
        except:
            print(i,j,k)
            print(index)
            print(cnode_data[i,j,k].shape, X_states.shape)
            raise Exception("")

    return cnode_data

def get_network_data(sim_data):
    state_mappings, ts, X_states = sim_data
    node_mappings, cnode_mappings = state_mappings

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

    for i in range(X_states.shape[1], model_dim):
        for oi in range(model_dim):
            network_data[:,oi] += X_states[:,i+oi]

    return network_data

def get_location_data(sim_data):
    state_mappings, ts, X_states = sim_data
    node_mappings, cnode_mappings = state_mappings

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

def get_community_data(sim_data):
    state_mappings, ts, X_states = sim_data
    node_mappings, cnode_mappings = state_mappings

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
        community_data[:,a,o,j] += X_states[:,node_mappings[a,o,i,j]]

    for a,o,i,j,k in cnode_mappings:
        community_data[:,a,o,j] += X_states[:,cnode_mappings[a,o,i,j,k]]

    return community_data

def get_full_array(self, X_state):
    full_X = np.zeros( (self.age_groups, self.model_dim, self.max_node_index+1, self.max_node_index+1) )
    full_cX = np.zeros( (self.age_groups, self.model_dim, self.max_node_index+1, self.max_node_index+1, self.max_node_index+1) )

    for ni in range(self.nodes_num):
        n = self.nodes[ni]
        for o in range(self.model_dim):
            full_X[n.age, o, n.home, n.loc] = X_state[n.state_index + o]

    for ni in range(self.cnodes_num):
        cn = self.cnodes[ni]
        for o in range(self.model_dim):
            full_cX[cn.age, o, cn.home, cn.fro, cn.to] = X_state[cn.state_index + o]

    return full_X, full_cX

def get_dt_schedule(times, end_time):
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