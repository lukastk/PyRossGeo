from libc.stdlib cimport malloc, free
import os
import numpy as np
cimport numpy as np
import scipy.special
import pandas as pd
import csv, json

from pyrossgeo.__defs__ cimport node, cnode, transporter, model_term, DTYPE_t
from pyrossgeo.__defs__ import DTYPE
from pyrossgeo.Simulation cimport Simulation

def initialize(self, sim_config_path='', model_dat='', commuter_networks_dat='',
                        node_parameters_dat='', cnode_parameters_dat='',
                        contact_matrices_dat='', node_cmatrices_dat='',
                        cnode_cmatrices_dat='', node_populations_dat='',
                        cnode_populations_dat=''):

    if self.has_been_initialized:
        raise Exception("Simulation has already been initialized.")

    #### Load data files

    default_commuter_networks_path = 'commuter_networks.csv'
    default_model_path = 'model.json'
    default_node_parameters_path = 'node_parameters.csv'
    default_cnode_parameters_path = 'cnode_parameters.csv'
    default_contact_matrices_path = 'contact_matrices.json'
    default_node_cmatrices_path = 'node_cmatrices.csv'
    default_cnode_cmatrices_path = 'cnode_cmatrices.csv'
    default_node_populations_path = 'node_populations.csv'
    default_cnode_populations_path = 'cnode_populations.csv'

    if type(commuter_networks_dat) == str:
        if commuter_networks_dat == '':
            commuter_networks_dat = os.path.join(sim_config_path, default_commuter_networks_path)
        commuter_networks_dat = pd.read_csv(commuter_networks_dat, delimiter=',', quotechar='"')

    if type(model_dat) == str:
        if model_dat == '':
            model_dat = os.path.join(sim_config_path, default_model_path)
        with open(model_dat, 'r') as json_file:
            model_dat = json.load(json_file)

    if type(node_parameters_dat) == str:
        if node_parameters_dat == '':
            node_parameters_dat = os.path.join(sim_config_path, default_node_parameters_path)
        node_parameters_dat = pd.read_csv(node_parameters_dat, delimiter=',', quotechar='"')
    elif type(node_parameters_dat) == dict:
        node_parameters_dat = pd.DataFrame(data=node_parameters_dat)

    if type(cnode_parameters_dat) == str:
        if cnode_parameters_dat == '':
            cnode_parameters_dat = os.path.join(sim_config_path, default_cnode_parameters_path)
        cnode_parameters_dat = pd.read_csv(cnode_parameters_dat, delimiter=',', quotechar='"')
    elif type(cnode_parameters_dat) == dict:
        cnode_parameters_dat = pd.DataFrame(data=cnode_parameters_dat)

    if type(contact_matrices_dat) == str:
        if contact_matrices_dat == '':
            contact_matrices_dat = os.path.join(sim_config_path, default_contact_matrices_path)
        with open(contact_matrices_dat, 'r') as json_file:
            contact_matrices_dat = json.load(json_file)
            for k in contact_matrices_dat:
                contact_matrices_dat[k] = np.array(contact_matrices_dat[k], dtype=DTYPE)

    if type(node_cmatrices_dat) == str:
        if node_cmatrices_dat == '':
            node_cmatrices_dat = os.path.join(sim_config_path, default_node_cmatrices_path)
        node_cmatrices_dat = pd.read_csv(node_cmatrices_dat, delimiter=',', quotechar='"')

    if type(cnode_cmatrices_dat) == str:
        if cnode_cmatrices_dat == '':
            cnode_cmatrices_dat = os.path.join(sim_config_path, default_cnode_cmatrices_path)
        cnode_cmatrices_dat = pd.read_csv(cnode_cmatrices_dat, delimiter=',', quotechar='"')

    if type(node_populations_dat) == str:
        if node_populations_dat == '':
            node_populations_dat = os.path.join(sim_config_path, default_node_populations_path)
        node_populations_dat = pd.read_csv(node_populations_dat, delimiter=',', quotechar='"')

    if type(cnode_populations_dat) == str:
        if cnode_populations_dat == '':
            cnode_populations_dat = os.path.join(sim_config_path, default_cnode_populations_path)
            if os.path.exists(cnode_populations_dat): # Avoid exceptions if user has not defined cnode_populations.csv
                cnode_populations_dat = pd.read_csv(cnode_populations_dat, delimiter=',', quotechar='"')
            else:
                cnode_populations_dat = None
        else:
            cnode_populations_dat = pd.read_csv(cnode_populations_dat, delimiter=',', quotechar='"')

    days_to_minutes = 1/(24*60.0) # Parameters are given in units of days, and need to be converted to minutes

    #### Find model_dim

    model_dim = len(model_dat['classes'])

    #### Find age_groups

    _cmat = contact_matrices_dat[ list(contact_matrices_dat.keys())[0] ]
    age_groups = _cmat.shape[0]

    #### Define variables

    max_node_index = -1 # The largest index of all nodes
    py_nodes = [] # List of nodes
    py_cnodes = [] # List of commuter nodes

    # Transport

    py_Ts = [] # List of transporters leading from origin to commuter node
    py_cTs = [] # List of transporters leading from commuter node to destination

    # Used for lambda calculation

    py_nodes_at_j = None # For each location j, and age bracket a, a list of the indices of nodes located at j.
    py_cnodes_into_k = None # For each location k, and age bracket a, a list of the indices of commuting nodes leading into k.

    # Arrays used during simulation

    aij_to_node = {} # Maps (age,home,location) to node
    aijk_to_cnode = {} # Maps (age,home,origin,destination) to cnode
    aijk_to_T = {} # Maps (age,home,origin,destination) to transporter between origin and commuting node
    aijk_to_cT = {} # Maps (age,home,origin,destination) to transporter commuting node and destination

    py_infection_classes_indices = [] # A list containing the class index of each infecting class.
    py_class_infections = [] # For each class index oi, a list of infection classes it is infected by
    py_linear_terms = [] # For each class index oi, a list of classes that oi interacts with linearly.

    py_contact_matrices_key_to_index = {} # Dictionary mapping keys to contact matrices
    py_contact_matrices = [] # A list of all contact matrices used

    py_location_area = None # Array containing the areas of each location

    #### Go through the commuter networks, and add nodes and cnodes ####

    for i, row in commuter_networks_dat.iterrows():
        home = int(row[0]) # commuter_networks_dat['Home']
        fro = int(row[1]) # commuter_networks_dat['From']
        to = int(row[2]) # commuter_networks_dat['To']
        age = int(row[3]) # commuter_networks_dat['Age']

        # The home node has not been defined yet, define it
        if not (age, home, home) in aij_to_node:
            home_node = py_node()
            home_node.node_index = len(py_nodes)
            home_node.home = home
            home_node.loc = home
            home_node.age = age
            py_nodes.append(home_node)
            aij_to_node[(home_node.age, home_node.home, home_node.loc)] = home_node
        home_node = aij_to_node[(age, home, home)]

        # Create transporter leading from origin to commuter node
        T = py_transporter()
        T.age = age
        T.home = home
        T.fro = fro
        T.to = to
        T.T_index = len(py_Ts)
        py_Ts.append(T)
        aijk_to_T[(age, home, fro, to)] = T

        # Create transporter leading from commuter node to destination
        cT = py_transporter()
        cT.age = age
        cT.home = home
        cT.fro = fro
        cT.to = to
        cT.T_index = len(py_cTs)
        py_cTs.append(cT)
        aijk_to_cT[(age, home, fro, to)] = cT

        move_N, move_percentage, t1, t2, ct1, ct2 = row[4:10]
        moving_classes = [k == 1 for k in row[10:]]
        use_percentage = move_percentage != -1

        if use_percentage and move_N != -1:
            raise Exception("Both move_N and move_percentage specified.")

        T.t1 = t1*60 # Convert units of time from hours to minutes
        T.t2 = t2*60 
        T.r_T_Delta_t = 1.0 / (T.t2 - T.t1)
        T.move_N = move_N
        T.move_percentage = move_percentage
        T.use_percentage = use_percentage
        T.moving_classes = moving_classes

        cT.t1 = ct1*60 
        cT.t2 = ct2*60 
        cT.r_T_Delta_t = 1.0 / (cT.t2 - cT.t1)
        cT.move_N = -1
        cT.move_percentage = 1.0 # All must leave the commuterverse
        cT.use_percentage = True
        cT.moving_classes = [True for i in range(model_dim)] # All classes can leave commuterverses

        # Create the origin-node

        if not (age, home, fro) in aij_to_node:
            node = py_node()
            node.node_index = len(py_nodes)
            node.home = home
            node.loc = fro
            node.age = age
            py_nodes.append(node)
            aij_to_node[(node.age, node.home, node.loc)] = node
        aij_to_node[(age, home, fro)].outgoing_T_indices.append(T.T_index)

        # Create the destination-node

        if not (age, home, to) in aij_to_node:
            node = py_node()
            node.node_index = len(py_nodes)
            node.home = home
            node.loc = to
            node.age = age
            py_nodes.append(node)
            aij_to_node[(node.age, node.home, node.loc)] = node
        aij_to_node[(age, home, to)].incoming_T_indices.append(cT.T_index)

        # Create the commuterverse linking the origin and the desintation node

        cnode = py_cnode()
        cnode.cnode_index = len(py_cnodes)
        cnode.home = home
        cnode.fro = fro
        cnode.to = to
        cnode.age = age
        cnode.incoming_node = aij_to_node[(age, home, fro)].node_index
        cnode.outgoing_node = aij_to_node[(age, home, to)].node_index
        cnode.incoming_T = T.T_index
        cnode.outgoing_T = cT.T_index
        py_cnodes.append(cnode)
        aijk_to_cnode[(cnode.age, cnode.home, cnode.fro, cnode.to)] = cnode

    #### Assign node indices to py_Ts ##################################

    for T in py_Ts:
        fro_node = aij_to_node[(T.age, T.home, T.fro)]
        to_node = aij_to_node[(T.age, T.home, T.to)]
        Tcnode = aijk_to_cnode[(T.age, T.home, T.fro, T.to)]
        T.fro_node_index = fro_node.node_index
        T.to_node_index = to_node.node_index
        T.cnode_index = Tcnode.cnode_index

    for cT in py_cTs:
        fro_node = aij_to_node[(cT.age, cT.home, cT.fro)]
        to_node = aij_to_node[(cT.age, cT.home, cT.to)]
        Tcnode = aijk_to_cnode[(cT.age, cT.home, cT.fro, cT.to)]
        cT.fro_node_index = fro_node.node_index
        cT.to_node_index = to_node.node_index
        cT.cnode_index = Tcnode.cnode_index

    #### Populations ###################################################

    for i, row in node_populations_dat.iterrows():
        home = int(row[0]) # row['Home']
        loc = int(row[1]) # row['Location']
        age_group_pops = np.array(row[2:]).reshape( (age_groups, model_dim) )

        for age in range(age_group_pops.shape[0]):
            state_pop = age_group_pops[age, :] 

            if not (age, home, loc) in aij_to_node:
                node = py_node()
                node.home = home
                node.loc = loc
                node.age = age
                node.node_index = len(py_nodes)
                aij_to_node[(age, home, loc)] = node
                py_nodes.append(node)

            node = aij_to_node[(age, home, loc)]
            node.state_pop = state_pop

    #### Commuterverse populations #####################################

    if not cnode_populations_dat is None:
        for i, row in cnode_populations_dat.iterrows():
            home = int(row['Home'])
            fro = int(row['From'])
            to = int(row['To'])
            age_group_pops = np.array(row[3:]).reshape( (age_groups, model_dim) )

            for age in range(age_group_pops.shape[0]):
                state_pop = age_group_pops[age, :] 
                cnode = aijk_to_cnode[age, home, fro, to]
                cnode.state_pop = state_pop

    #### Find max_node_index ###########################################

    for a,i,j in aij_to_node:
        if i > max_node_index:
            max_node_index = i
        if j > max_node_index:
            max_node_index = j

    ##### NumPy-ify the node fields and assign each node ###############
    ##### an index of the state vector                   ###############

    current_X_state_index = 0

    for node in py_nodes:
        node.incoming_T_indices = np.array(node.incoming_T_indices, dtype=int)
        node.outgoing_T_indices = np.array(node.outgoing_T_indices, dtype=int)
        node.state_index = current_X_state_index
        current_X_state_index += model_dim

    node_states_len = current_X_state_index

    for cnode in py_cnodes:
        cnode.state_index = current_X_state_index
        current_X_state_index += model_dim

    #### Set the initial conditions ####################################

    X_state0 = np.zeros(current_X_state_index)

    for node in py_nodes:
        if not node.state_pop is None:
            X_state0[node.state_index:node.state_index+model_dim] = node.state_pop

    for cnode in py_cnodes:
        if not cnode.state_pop is None:
            X_state0[cnode.state_index:cnode.state_index+model_dim] = cnode.state_pop

    #### Create py_nodes_at_j ##########################################

    py_nodes_at_j = [ [ [] for j in range(max_node_index+1) ] for a in range(age_groups)]

    for j in range(max_node_index+1):
        for age, home, loc_j in aij_to_node:
            if loc_j == j:
                py_nodes_at_j[age][loc_j].append(aij_to_node[(age, home, loc_j)].node_index )

    for a in range(len(py_nodes_at_j)):
        for j in range(len(py_nodes_at_j[a])):
            py_nodes_at_j[a][j] = np.array(py_nodes_at_j[a][j], dtype=int)

    #### Create py_cnodes_into_k #######################################

    py_cnodes_into_k = [ [ [] for j in range(max_node_index+1) ] for a in range(age_groups)]

    for k in range(max_node_index+1):
        for age, home, fro, to in aijk_to_cnode:
            if to == k:
                py_cnodes_into_k[age][to].append(aijk_to_cnode[(age, home, fro, to)].cnode_index)

    for a in range(len(py_cnodes_into_k)):
        for k in range(len(py_cnodes_into_k[a])):
            py_cnodes_into_k[a][k] = np.array(py_cnodes_into_k[a][k], dtype=int)

    #### Generate state mapping dictionaries ###########################

    node_state_index_mappings = {}
    cnode_state_index_mappings = {}

    for n in py_nodes:
        for o in range(model_dim):
            node_state_index_mappings[n.age, o, n.home, n.loc] = n.state_index+o

    for cn in py_cnodes:
        for o in range(model_dim):
            cnode_state_index_mappings[cn.age, o, cn.home, cn.fro, cn.to] = cn.state_index+o

    py_state_mappings = (node_state_index_mappings, cnode_state_index_mappings)

    #### Set node and cnode parameters #################################

    # Note: This section is the most complicated part of the initialisation procedure.

    model_class_name_to_class_index = {}
    model_class_index_to_class_name = {}
    for i in range(len(model_dat['classes'])):
        oclass = model_dat['classes'][i]
        model_class_name_to_class_index[oclass] = i
        model_class_index_to_class_name[i] = oclass

    py_model_linear_terms = []
    py_model_infection_terms = []

    infection_model_param_to_model_term = {}
    linear_model_param_to_model_term = {}

    ## Construct internal representation of model

    for class_name in model_dat:
        if class_name == 'classes':
            continue

        for coupling_class, model_param in model_dat[class_name]['linear']:
            if model_param[0] == '-':
                is_neg = True
                model_param = model_param[1:]
            else:
                is_neg = False

            if not model_param in linear_model_param_to_model_term:
                mt = py_model_term()
                mt.model_param = model_param
                linear_model_param_to_model_term[model_param] = mt
                py_model_linear_terms.append(mt)
            mt = linear_model_param_to_model_term[model_param]
            if is_neg:
                mt.oi_neg = model_class_name_to_class_index[class_name]
            else:
                mt.oi_pos = model_class_name_to_class_index[class_name]
            mt.oi_coupling = model_class_name_to_class_index[coupling_class]

        for coupling_class, model_param in model_dat[class_name]['infection']: 
            if model_param[0] == '-':
                is_neg = True
                model_param = model_param[1:]
            else:
                is_neg = False

            if not model_param in infection_model_param_to_model_term:
                mt = py_model_term()
                mt.model_param = model_param
                infection_model_param_to_model_term[model_param] = mt
                py_model_infection_terms.append(mt)
            mt = infection_model_param_to_model_term[model_param]
            if is_neg:
                mt.oi_neg = model_class_name_to_class_index[class_name]
            else:
                mt.oi_pos = model_class_name_to_class_index[class_name]
            mt.oi_coupling = model_class_name_to_class_index[coupling_class]

    # Find all infection classes (py_infection_classes_indices), and assign model_term.infection_index
    
    for model_param in linear_model_param_to_model_term:
        mt = linear_model_param_to_model_term[model_param]
    
    for model_param in infection_model_param_to_model_term:
        mt = infection_model_param_to_model_term[model_param]

        if not mt.oi_coupling in infection_model_param_to_model_term:
            py_infection_classes_indices.append(mt.oi_coupling)

        mt.infection_index = py_infection_classes_indices.index(mt.oi_coupling)

    py_infection_classes_indices = np.array(py_infection_classes_indices, dtype=np.dtype('i'))

    ## Set node model parameters

    py_location_area = np.zeros( max_node_index+1 )

    for n in py_nodes:
        for row_i, row in node_parameters_dat.iterrows():
            home = row[0] # row['Home']
            loc = row[1] # row['Location']
            age = row[2] # row['Age']
            area = row[3] # row['Area']

            home = int(home)  if home != 'ALL' else 'ALL'
            loc = int(loc) if (loc != 'ALL' and loc != 'HOME') else loc
            age = int(age)  if age != 'ALL' else 'ALL'

            if not (home == n.home or home == 'ALL'):
                continue
            if not (loc == n.loc or loc == 'ALL' or (n.loc==n.home and loc=='HOME')):
                continue
            if not (age == n.age or age == 'ALL'):
                continue

            if n.home == n.loc: # Areas are only defined for locations, not specific nodes
                py_location_area[n.loc] = area

            for i in range(len(py_model_linear_terms)):
                mt = py_model_linear_terms[i]
                param_val = DTYPE(row[mt.model_param]* days_to_minutes) 
                n.linear_coeffs.append(param_val)

            for i in range(len(py_model_infection_terms)):
                mt = py_model_infection_terms[i]
                param_val = DTYPE(row[mt.model_param]* days_to_minutes) 
                n.infection_coeffs.append(param_val)

    ## Set cnode model parameters

    for cn in py_cnodes:
        for row_i, row in cnode_parameters_dat.iterrows():
            home = row[0]# row['Home']
            fro = row[1]# row['From']
            to = row[2]# row['To']
            age = row[3]# row['Age']
            area = row[4]# row['Area']
            home = int(home)  if home != 'ALL' else 'ALL'
            fro = int(fro) if (fro != 'ALL' and fro != 'HOME') else fro
            to = int(to) if (to != 'ALL' and to != 'HOME') else to
            age = int(age)  if age != 'ALL' else 'ALL'
        
            if not (home == cn.home or home == 'ALL'):
                continue
            if not (fro == cn.fro or fro == 'ALL' or (cn.fro==cn.home and fro=='HOME')):
                continue
            if not (to == cn.to or to == 'ALL' or (cn.to==cn.home and to=='HOME')):
                continue
            if not (age == cn.age or age == 'ALL'):
                continue

            cn.area = area

            for i in range(len(py_model_linear_terms)):
                mt = py_model_linear_terms[i]
                param_val = DTYPE(row[mt.model_param]* days_to_minutes) 
                cn.linear_coeffs.append(param_val)

            for i in range(len(py_model_infection_terms)):
                mt = py_model_infection_terms[i]
                param_val = DTYPE(row[mt.model_param]* days_to_minutes) 
                cn.infection_coeffs.append(param_val)

    #### Set contact matrices ##########################################

    for cmat_key in contact_matrices_dat:
        py_contact_matrices_key_to_index[ cmat_key ] = len(py_contact_matrices)
        py_contact_matrices.append( contact_matrices_dat[cmat_key] )

    py_contact_matrices = np.array(py_contact_matrices, dtype=DTYPE)
    
    # Set node contact matrices

    for row_i, row in node_cmatrices_dat.iterrows():
        home = row[0] # row['Home']
        loc = row[1] # row['Location']
        home = int(home) if home != 'ALL' else home
        loc = int(loc) if (loc != 'ALL' and loc != 'HOME') else loc
            
        cmat_indices = np.array([py_contact_matrices_key_to_index[ckey] if ckey in py_contact_matrices_key_to_index else -1 for ckey in row[2:]])
        
        for home_i in range(max_node_index+1):
            _loc = home_i if loc == 'HOME' else loc

            for loc_j in range(max_node_index+1):
                if (home_i == home or home == 'ALL') and (loc_j == _loc or _loc == 'ALL'):
                    for age_a in range(age_groups):
                        n = aij_to_node.get((age_a, home_i, loc_j))
                        if not n is None:
                            n.contact_matrix_indices = cmat_indices[py_infection_classes_indices]
                            if -1 in n.contact_matrix_indices:
                                raise Exception("Contact matrix missing for node (%i, %i, %i)" % (age_a, home_i, loc_j))

    # Set cnode contact matrices

    for row_i, row in cnode_cmatrices_dat.iterrows():
        home = row[0] #row['Home']
        fro = row[1] #row['From']
        to = row[2] #row['To']
        home = int(home) if home != 'ALL' else home
        fro = int(loc) if (fro != 'ALL' and fro != 'HOME') else fro
        to = int(to) if (to != 'ALL' and to != 'HOME') else to

        cmat_indices = np.array([py_contact_matrices_key_to_index[ckey] if ckey in py_contact_matrices_key_to_index else -1 for ckey in row[3:]])

        for home_i in range(max_node_index+1):
            _fro = home_i if fro == 'HOME' else fro
            _to = home_i if to == 'HOME' else to

            for fro_j in range(max_node_index+1):
                for to_j in range(max_node_index+1):
                    if (home_i == home or home == 'ALL') and (fro_j == _fro or _fro == 'ALL') and (to_j == _to or _to == 'ALL'):
                        for age_a in range(age_groups):
                            cn = aijk_to_cnode.get((age_a, home_i, fro_j, to_j))
                            if not cn is None:
                                cn.contact_matrix_indices = cmat_indices[py_infection_classes_indices]

                                if -1 in cn.contact_matrix_indices:
                                    raise Exception("Contact matrix missing for cnode (%i, %i, %i)" % (age_a, home_i, fro_j, to_j))

    return _initialize(self, max_node_index, model_dim, age_groups, X_state0, node_states_len, py_nodes, py_cnodes,
                        py_Ts, py_cTs, py_nodes_at_j, py_cnodes_into_k, py_state_mappings,
                        py_infection_classes_indices, py_model_linear_terms, py_model_infection_terms,
                        py_contact_matrices, py_contact_matrices_key_to_index)


cdef _initialize(Simulation self, py_max_node_index, model_dim, age_groups, X_state_arr, node_states_len, py_nodes, py_cnodes,
                        py_Ts, py_cTs, py_nodes_at_j, py_cnodes_into_k, py_state_mappings,
                        py_infection_classes_indices, py_model_linear_terms, py_model_infection_terms,
                        py_contact_matrices, py_contact_matrices_key_to_index):
    """Initialize the simulation."""

    self.has_been_initialized = True

    self.node_mappings, self.cnode_mappings = py_state_mappings

    # Initialize the transport profile
    # TODO this should go in a config file
    self.transport_profile_c = 0.1
    self.transport_profile_c_r = 1 / (2 * self.transport_profile_c * self.transport_profile_c)
    self.transport_profile_integrated = self.transport_profile_c * np.sqrt(2 * np.pi) * scipy.special.erf( 1 / (2 * np.sqrt(2) * self.transport_profile_c) )
    self.transport_profile_integrated_r = 1/self.transport_profile_integrated
    self.transport_profile_m = 0.5

    self.model_dim = model_dim
    self.max_node_index = py_max_node_index
    self.age_groups = age_groups
    self.node_states_len = node_states_len

    cdef DTYPE_t[:] X_state = X_state_arr

    self.nodes_num = len(py_nodes)
    self.nodes = <node *> malloc(self.nodes_num * sizeof(node))
    for ni in range(self.nodes_num):
        pn = py_nodes[ni]
        self.nodes[ni].home = pn.home
        self.nodes[ni].loc = pn.loc
        self.nodes[ni].age = pn.age
        self.nodes[ni].state_index = pn.state_index

        self.nodes[ni].contact_matrix_indices = <int *> malloc(len(pn.contact_matrix_indices) * sizeof(int))
        for i in range(len(pn.contact_matrix_indices)):
            self.nodes[ni].contact_matrix_indices[i] = pn.contact_matrix_indices[i]

        self.nodes[ni].incoming_T_indices = <int *> malloc(len(pn.incoming_T_indices) * sizeof(int))
        self.nodes[ni].incoming_T_indices_len = len(pn.incoming_T_indices)
        self.nodes[ni].outgoing_T_indices = <int *> malloc(len(pn.outgoing_T_indices) * sizeof(int))
        self.nodes[ni].outgoing_T_indices_len = len(pn.outgoing_T_indices)

        for Ti in range(self.nodes[ni].incoming_T_indices_len):
            self.nodes[ni].incoming_T_indices[Ti] = pn.incoming_T_indices[Ti]
        for Ti in range(self.nodes[ni].outgoing_T_indices_len):
            self.nodes[ni].outgoing_T_indices[Ti] = pn.outgoing_T_indices[Ti]

        self.nodes[ni].linear_coeffs = <DTYPE_t *> malloc(len(pn.linear_coeffs) * sizeof(DTYPE_t))
        for i in range(len(pn.linear_coeffs)):
            self.nodes[ni].linear_coeffs[i] = pn.linear_coeffs[i]

        self.nodes[ni].infection_coeffs = <DTYPE_t *> malloc(len(pn.infection_coeffs) * sizeof(DTYPE_t))
        for i in range(len(pn.infection_coeffs)):
            self.nodes[ni].infection_coeffs[i] = pn.infection_coeffs[i]

    self.cnodes_num = len(py_cnodes)
    self.cnodes = <cnode *> malloc(self.cnodes_num * sizeof(cnode))
    for ni in range(self.cnodes_num):
        pcn = py_cnodes[ni]
        self.cnodes[ni].home = pcn.home
        self.cnodes[ni].fro = pcn.fro
        self.cnodes[ni].to = pcn.to
        self.cnodes[ni].age = pcn.age
        self.cnodes[ni].state_index = pcn.state_index
        self.cnodes[ni].incoming_node = pcn.incoming_node
        self.cnodes[ni].outgoing_node = pcn.outgoing_node
        self.cnodes[ni].incoming_T = pcn.incoming_T
        self.cnodes[ni].outgoing_T = pcn.outgoing_T
        self.cnodes[ni].is_on = False
        self.cnodes[ni].area = pcn.area

        self.cnodes[ni].contact_matrix_indices = <int *> malloc(len(pcn.contact_matrix_indices) * sizeof(int))
        for i in range(len(pcn.contact_matrix_indices)):
            self.cnodes[ni].contact_matrix_indices[i] = pcn.contact_matrix_indices[i]

        self.cnodes[ni].linear_coeffs = <DTYPE_t *> malloc(len(pcn.linear_coeffs) * sizeof(DTYPE_t))
        for i in range(len(pcn.linear_coeffs)):
            self.cnodes[ni].linear_coeffs[i] = pcn.linear_coeffs[i]

        self.cnodes[ni].infection_coeffs = <DTYPE_t *> malloc(len(pcn.infection_coeffs) * sizeof(DTYPE_t))
        for i in range(len(pcn.infection_coeffs)):
            self.cnodes[ni].infection_coeffs[i] = pcn.infection_coeffs[i]

    # Transport

    self.Ts_num = len(py_Ts)
    self.Ts = <transporter *> malloc(self.Ts_num * sizeof(transporter))
    for ti in range(self.Ts_num):
        self.Ts[ti].T_index = py_Ts[ti].T_index
        self.Ts[ti].age = py_Ts[ti].age
        self.Ts[ti].home = py_Ts[ti].home
        self.Ts[ti].fro = py_Ts[ti].fro
        self.Ts[ti].to = py_Ts[ti].to
        self.Ts[ti].fro_node_index = py_Ts[ti].fro_node_index
        self.Ts[ti].to_node_index = py_Ts[ti].to_node_index
        self.Ts[ti].cnode_index = py_Ts[ti].cnode_index
        self.Ts[ti].is_on = False
        
        self.Ts[ti].t1 = py_Ts[ti].t1
        self.Ts[ti].t2 = py_Ts[ti].t2
        self.Ts[ti].r_T_Delta_t = py_Ts[ti].r_T_Delta_t
        self.Ts[ti].move_N = py_Ts[ti].move_N
        self.Ts[ti].move_percentage = py_Ts[ti].move_percentage
        self.Ts[ti].use_percentage = py_Ts[ti].use_percentage
        self.Ts[ti].moving_classes = <bint *> malloc(self.model_dim * sizeof(bint))
        for j in range(self.model_dim):
            self.Ts[ti].moving_classes[j] = py_Ts[ti].moving_classes[j]

    self.cTs_num = len(py_cTs)
    self.cTs = <transporter *> malloc(self.cTs_num * sizeof(transporter))
    for ti in range(self.cTs_num):
        self.cTs[ti].T_index = py_cTs[ti].T_index
        self.cTs[ti].age = py_cTs[ti].age
        self.cTs[ti].home = py_cTs[ti].home
        self.cTs[ti].fro = py_cTs[ti].fro
        self.cTs[ti].to = py_cTs[ti].to
        self.cTs[ti].fro_node_index = py_cTs[ti].fro_node_index
        self.cTs[ti].to_node_index = py_cTs[ti].to_node_index
        self.cTs[ti].cnode_index = py_cTs[ti].cnode_index
        self.cTs[ti].is_on = False

        self.cTs[ti].t1 = py_cTs[ti].t1
        self.cTs[ti].t2 = py_cTs[ti].t2
        self.cTs[ti].r_T_Delta_t = py_cTs[ti].r_T_Delta_t
        self.cTs[ti].move_N = py_cTs[ti].move_N
        self.cTs[ti].move_percentage = py_cTs[ti].move_percentage
        self.cTs[ti].use_percentage = py_cTs[ti].use_percentage
        self.cTs[ti].moving_classes = <bint *> malloc(self.model_dim * sizeof(bint))
        for j in range(self.model_dim):
            self.cTs[ti].moving_classes[j] = py_cTs[ti].moving_classes[j]

    # F

    self.nodes_at_j = <int ***> malloc(self.age_groups * sizeof(int **))
    self.nodes_at_j_len = <int **> malloc(self.age_groups * sizeof(int *))
    for age in range(self.age_groups):
        self.nodes_at_j[age] = <int **> malloc( len(py_nodes_at_j[age]) * sizeof(int *))
        self.nodes_at_j_len[age] = <int *> malloc( len(py_nodes_at_j[age]) * sizeof(int))
        for loc in range(len(py_nodes_at_j[age])):
            self.nodes_at_j[age][loc] = <int *> malloc( len(py_nodes_at_j[age][loc]) * sizeof(int))
            self.nodes_at_j_len[age][loc] = len(py_nodes_at_j[age][loc])
            for i in range(len(py_nodes_at_j[age][loc])):
                self.nodes_at_j[age][loc][i] = py_nodes_at_j[age][loc][i]

    # G

    self.cnodes_into_k = <int ***> malloc(self.age_groups * sizeof(int **))
    self.cnodes_into_k_len = <int **> malloc(self.age_groups * sizeof(int *))
    for age in range(self.age_groups):
        self.cnodes_into_k[age] = <int **> malloc( len(py_cnodes_into_k[age]) * sizeof(int *))
        self.cnodes_into_k_len[age] = <int *> malloc( len(py_cnodes_into_k[age]) * sizeof(int))
        for loc in range(len(py_cnodes_into_k[age])):
            self.cnodes_into_k[age][loc] = <int *> malloc( len(py_cnodes_into_k[age][loc]) * sizeof(int))
            self.cnodes_into_k_len[age][loc] = len(py_cnodes_into_k[age][loc])
            for k in range(len(py_cnodes_into_k[age][loc])):
                self.cnodes_into_k[age][loc][k] = py_cnodes_into_k[age][loc][k]

    self.state_size = len(X_state_arr)

    # Model

    self.infection_classes_indices = py_infection_classes_indices
    self.infection_classes_num = len(py_infection_classes_indices)

    self.model_linear_terms = <model_term *> malloc(len(py_model_linear_terms) * sizeof(model_term))
    self.model_linear_terms_len = len(py_model_linear_terms)
    for i in range(len(py_model_linear_terms)):
        py_mt = py_model_linear_terms[i]
        self.model_linear_terms[i].oi_pos = py_mt.oi_pos
        self.model_linear_terms[i].oi_neg = py_mt.oi_neg
        self.model_linear_terms[i].oi_coupling = py_mt.oi_coupling
        self.model_linear_terms[i].infection_index = py_mt.infection_index

    self.model_infection_terms = <model_term *> malloc(len(py_model_infection_terms) * sizeof(model_term))
    self.model_infection_terms_len = len(py_model_infection_terms)
    for i in range(len(py_model_infection_terms)):
        py_mt = py_model_infection_terms[i]
        self.model_infection_terms[i].oi_pos = py_mt.oi_pos
        self.model_infection_terms[i].oi_neg = py_mt.oi_neg
        self.model_infection_terms[i].oi_coupling = py_mt.oi_coupling
        self.model_infection_terms[i].infection_index = py_mt.infection_index

    self.contact_matrices = py_contact_matrices
    self.contact_matrices_key_to_index = py_contact_matrices_key_to_index

    self._lambdas = np.zeros( (self.contact_matrices.shape[0], self.age_groups, self.infection_classes_num) )

    self._Is_arr = np.zeros( (self.age_groups) )
    self._Is = self._Is_arr

    self._Ns_arr = np.zeros( (self.age_groups) )
    self._Ns = self._Ns_arr

    return X_state_arr

class py_node:
    def __init__(self):
        self.node_index = -1
        self.home = -1
        self.loc = -1
        self.age = -1
        self.state_index = -1
        self.incoming_T_indices = []
        self.outgoing_T_indices = []
        self.state_pop = None
        self.linear_coeffs = []
        self.infection_coeffs = []
        self.contact_matrix_indices = None

    def __str__(self):
        return "ni: %s, Age: %s, Home: %s, Loc: %s" % (self.node_index, self.age, self.home, self.loc)

class py_cnode:
    def __init__(self):
        self.cnode_index = -1
        self.home = -1
        self.fro = -1
        self.to = -1
        self.age = -1
        self.state_index = -1
        self.incoming_node = -1
        self.outgoing_node = -1
        self.incoming_T = -1
        self.outgoing_T = -1
        self.area = -1
        self.state_pop = None
        self.linear_coeffs = []
        self.infection_coeffs = []
        self.is_on = False
        self.contact_matrix_indices = None

    def __str__(self):
        return "cni: %s, Age: %s, Home: %s, Fro: %s, To: %s" % (self.cnode_index, self.age, self.home, self.fro, self.to)

class py_transporter:
    def __init__(self):
        self.T_index = -1
        self.age = -1
        self.home = -1
        self.fro = -1
        self.fro_node_index = -1
        self.to = -1
        self.to_node_index = -1
        self.cnode_index = -1
        self.t1 = -1
        self.t2 = -1
        self.r_T_Delta_t = -1
        self.move_N = -1
        self.move_percentage = -1
        self.use_percentage = -1
        self.moving_classes = -1

    def __str__(self):
        return "ti: %s, a: %s, home: %s, fro: %s, to: %s" % (self.T_index, self.age, self.home, self.fro, self.to)

class py_model_term:
    def __init__(self):
        self.oi_pos = -1
        self.oi_neg = -1
        self.oi_coupling  = -1
        self.infection_index = -1