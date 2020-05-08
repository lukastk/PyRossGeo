from libc.stdlib cimport malloc, free
import os
import numpy as np
cimport numpy as np
import scipy.special
import pandas as pd
import csv, json

from pyrossgeo.__defs__ cimport node, cnode, transporter, DTYPE_t
from pyrossgeo.__defs__ import DTYPE
from pyrossgeo.Simulation cimport Simulation

def initialize(self, sim_data_path='', model_dat='', commuter_networks_dat='',
                        node_parameters_dat='', cnode_parameters_dat='',
                        contact_matrices_dat='', node_cmatrices_dat='',
                        cnode_cmatrices_dat='', node_populations_dat='',
                        cnode_populations_dat=''):

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
            commuter_networks_dat = os.path.join(sim_data_path, default_commuter_networks_path)
        print(sim_data_path, commuter_networks_dat)
        commuter_networks_dat = pd.read_csv(commuter_networks_dat, delimiter=',', quotechar='"')

    if type(model_dat) == str:
        if model_dat == '':
            model_dat = os.path.join(sim_data_path, default_model_path)
        with open(model_dat, 'r') as json_file:
            model_dat = json.load(json_file)

    if type(node_parameters_dat) == str:
        if node_parameters_dat == '':
            node_parameters_dat = os.path.join(sim_data_path, default_node_parameters_path)
        node_parameters_dat = pd.read_csv(node_parameters_dat, delimiter=',', quotechar='"')
    elif type(node_parameters_dat) == dict:
        node_parameters_dat = pd.DataFrame(data=node_parameters_dat)

    if type(cnode_parameters_dat) == str:
        if cnode_parameters_dat == '':
            cnode_parameters_dat = os.path.join(sim_data_path, default_cnode_parameters_path)
        cnode_parameters_dat = pd.read_csv(cnode_parameters_dat, delimiter=',', quotechar='"')
    elif type(cnode_parameters_dat) == dict:
        cnode_parameters_dat = pd.DataFrame(data=cnode_parameters_dat)

    if type(contact_matrices_dat) == str:
        if contact_matrices_dat == '':
            contact_matrices_dat = os.path.join(sim_data_path, default_contact_matrices_path)
        with open(contact_matrices_dat, 'r') as json_file:
            contact_matrices_dat = json.load(json_file)
            for k in contact_matrices_dat:
                contact_matrices_dat[k] = np.array(contact_matrices_dat[k], dtype=DTYPE)

    if type(node_cmatrices_dat) == str:
        if node_cmatrices_dat == '':
            node_cmatrices_dat = os.path.join(sim_data_path, default_node_cmatrices_path)
        node_cmatrices_dat = pd.read_csv(node_cmatrices_dat, delimiter=',', quotechar='"')

    if type(cnode_cmatrices_dat) == str:
        if cnode_cmatrices_dat == '':
            cnode_cmatrices_dat = os.path.join(sim_data_path, default_cnode_cmatrices_path)
        cnode_cmatrices_dat = pd.read_csv(cnode_cmatrices_dat, delimiter=',', quotechar='"')

    if type(node_populations_dat) == str:
        if node_populations_dat == '':
            node_populations_dat = os.path.join(sim_data_path, default_node_populations_path)
        node_populations_dat = pd.read_csv(node_populations_dat, delimiter=',', quotechar='"')

    if type(cnode_populations_dat) == str:
        if cnode_populations_dat == '':
            cnode_populations_dat = os.path.join(sim_data_path, default_cnode_populations_path)
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

    model_class_to_class_index_o = {}
    model_class_index_o_to_class = {}
    for i in range(len(model_dat['classes'])):
        oclass = model_dat['classes'][i]
        model_class_to_class_index_o[oclass] = i
        model_class_index_o_to_class[i] = oclass

    # Infection terms

    model_class_to_infection_class_index = {}
    model_parameter_to_class_index = {}
    for oclass in model_dat:
        if oclass == 'classes':
            continue
        for uclass, model_parameter_key in model_dat[oclass]['infection']:
            class_index = model_class_to_class_index_o[uclass]
            if not class_index in py_infection_classes_indices:
                model_class_to_infection_class_index[ uclass ] = len(py_infection_classes_indices)
                py_infection_classes_indices.append( int(class_index) )
            if not model_parameter_key in model_parameter_to_class_index:
                model_parameter_to_class_index[model_parameter_key] = model_class_to_class_index_o[uclass]

    py_infection_classes_indices = np.array(py_infection_classes_indices, dtype=np.dtype("i"))
    
    infection_terms_class_index_to_index = []
    for o in range(model_dim):
        py_class_infections.append( [] )
        infection_terms_class_index_to_index.append( {} )
        for oclass, _ in model_dat[ model_class_index_o_to_class[o] ]['infection']:
            infection_terms_class_index_to_index[o][ model_class_to_class_index_o[oclass] ] = len(py_class_infections[o])
            py_class_infections[o].append( model_class_to_infection_class_index[oclass] )

    # Linear terms
 
    for o in range(model_dim):
        py_linear_terms.append([])
        for oclass, model_parameter_key in model_dat[ model_class_index_o_to_class[o] ]['linear']:
            py_linear_terms[o].append( model_class_to_class_index_o[oclass] )

            if not model_parameter_key in model_parameter_to_class_index:
                model_parameter_to_class_index[model_parameter_key] = model_class_to_class_index_o[oclass]

    linear_terms_class_index_to_index = []
    for o in range(model_dim):
        linear_terms_class_index_to_index.append({})
        for i in range(len(py_linear_terms[o])):
            linear_terms_class_index_to_index[o][ py_linear_terms[o][i] ] = i

    # Set the model parameters for each node

    py_location_area = np.zeros( max_node_index+1 )

    for n in py_nodes:
        for row_i, row in node_parameters_dat.iterrows():
            home = row[0] # row['Home']
            loc = row[1] # row['Location']
            age = row[2] # row['Age']
            area = row[3] # row['Area']

            home = int(home)  if home != 'ALL' else 'ALL'
            age = int(age)  if age != 'ALL' else 'ALL'

            if loc == 'HOME':
                loc = home
            elif loc != 'ALL':
                loc = int(loc)

            if not (home == n.home or home == 'ALL'):
                continue
            if not (loc == n.loc or loc == 'ALL'):
                continue
            if not (age == n.age or age == 'ALL'):
                continue

            if n.home == n.loc: # Areas are only defined for locations, not specific nodes
                py_location_area[n.loc] = area

            n.linear_coeffs = [ np.zeros( len(linear_terms_class_index_to_index[u]) ) for u in range(model_dim) ]
            n.infection_coeffs = [ np.zeros( len(infection_terms_class_index_to_index[u]) ) for u in range(model_dim) ]

            for oclass in model_dat:
                if oclass == 'classes':
                    continue
                
                o = model_class_to_class_index_o[oclass]
                for linear_class, param_key in model_dat[oclass]['linear']:
                    lo = model_class_to_class_index_o[linear_class]
                    if param_key[0] == '-':
                        param = -DTYPE(row[param_key[1:]]) * days_to_minutes
                    else:
                        param = DTYPE(row[param_key]) * days_to_minutes
                    n.linear_coeffs[o][linear_terms_class_index_to_index[o][lo]] = param

                for infected_class, param_key in model_dat[oclass]['infection']:
                    io = model_class_to_class_index_o[infected_class]
                    if param_key[0] == '-':
                        param = -DTYPE(row[param_key[1:]]) * days_to_minutes
                    else:
                        param = DTYPE(row[param_key]) * days_to_minutes
                    n.infection_coeffs[o][infection_terms_class_index_to_index[o][io]] = param

    # Set the model parameters for each commuter node

    for cn in py_cnodes:
        for row_i, row in cnode_parameters_dat.iterrows():
            home = row[0]# row['Home']
            fro = row[1]# row['From']
            to = row[2]# row['To']
            age = row[3]# row['Age']
            area = row[4]# row['Area']
            home = int(home)  if home != 'ALL' else 'ALL'
            age = int(age)  if age != 'ALL' else 'ALL'

            if fro == 'HOME':
                fro = home
            elif fro != 'ALL':
                fro = int(fro)
            if to == 'HOME':
                to = home
            elif to != 'ALL':
                to = int(to)

            if not (home == cn.home or home == 'ALL'):
                continue
            if not (fro == cn.fro or fro == 'ALL'):
                continue
            if not (to == cn.to or to == 'ALL'):
                continue
            if not (age == cn.age or age == 'ALL'):
                continue

            cn.area = area

            cn.linear_coeffs = [ np.zeros( len(linear_terms_class_index_to_index[u]) ) for u in range(model_dim) ]
            cn.infection_coeffs = [ np.zeros( len(infection_terms_class_index_to_index[u]) ) for u in range(model_dim) ]

            for oclass in model_dat:
                if oclass == 'classes':
                    continue
                
                o = model_class_to_class_index_o[oclass]
                
                for linear_class, param_key in model_dat[oclass]['linear']:
                    lo = model_class_to_class_index_o[linear_class]
                    if param_key[0] == '-':
                        param = -DTYPE(row[param_key[1:]]) * days_to_minutes
                    else:
                        param = DTYPE(row[param_key]) * days_to_minutes
                    cn.linear_coeffs[o][linear_terms_class_index_to_index[o][lo]] = param

                for infected_class, param_key in model_dat[oclass]['infection']:
                    io = model_class_to_class_index_o[infected_class]
                    if param_key[0] == '-':
                        param = -DTYPE(row[param_key[1:]]) * days_to_minutes
                    else:
                        param = DTYPE(row[param_key]) * days_to_minutes
                    cn.infection_coeffs[o][infection_terms_class_index_to_index[o][io]] = param

    #### Set contact matrices ##########################################

    for cmat_key in contact_matrices_dat:
        py_contact_matrices_key_to_index[ cmat_key ] = len(py_contact_matrices)
        py_contact_matrices.append( contact_matrices_dat[cmat_key] )

    py_contact_matrices = np.array(py_contact_matrices, dtype=DTYPE)
    
    # Set node contact matrices

    for row_i, row in node_cmatrices_dat.iterrows():
        home = row[0] # row['Home']
        loc = row[1] # row['Location']
        home = int(home) if home != 'ALL' else 'ALL'

        if loc == 'HOME':
            loc = home
        elif loc != 'ALL':
            loc = int(loc)
            
        cmat_indices = np.array([py_contact_matrices_key_to_index[ckey] if ckey in py_contact_matrices_key_to_index else -1 for ckey in row[2:]])

        for home_i in range(max_node_index+1):
            for loc_j in range(max_node_index+1):
                if (home_i == home or home == 'ALL') and (loc_j == loc or loc == 'ALL'):
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
        home = int(home) if home != 'ALL' else 'ALL'

        if fro == 'HOME':
            fro = home
        elif fro != 'ALL':
            fro = int(fro)
        if to == 'HOME':
            to = home
        elif to != 'ALL':
            to = int(to)

        cmat_indices = np.array([py_contact_matrices_key_to_index[ckey] if ckey in py_contact_matrices_key_to_index else -1 for ckey in row[3:]])

        for home_i in range(max_node_index+1):
            for fro_j in range(max_node_index+1):
                for to_j in range(max_node_index+1):
                    if (home_i == home or home == 'ALL') and (fro_j == fro or fro == 'ALL') and (to_j == to or to == 'ALL'):
                        for age_a in range(age_groups):
                            cn = aijk_to_cnode.get((age_a, home_i, fro_j, to_j))
                            if not cn is None:
                                cn.contact_matrix_indices = cmat_indices[py_infection_classes_indices]

                                if -1 in cn.contact_matrix_indices:
                                    raise Exception("Contact matrix missing for cnode (%i, %i, %i)" % (age_a, home_i, fro_j, to_j))

    return _initialize(self, max_node_index, model_dim, age_groups, X_state0, node_states_len, py_nodes, py_cnodes,
                        py_Ts, py_cTs, py_nodes_at_j, py_cnodes_into_k, py_state_mappings,
                        py_infection_classes_indices, py_class_infections, py_linear_terms,
                        py_contact_matrices, py_contact_matrices_key_to_index)


cdef _initialize(Simulation self, py_max_node_index, model_dim, age_groups, X_state_arr, node_states_len, py_nodes, py_cnodes,
                        py_Ts, py_cTs, py_nodes_at_j, py_cnodes_into_k, py_state_mappings,
                        py_infection_classes_indices, py_class_infections, py_linear_terms,
                        py_contact_matrices, py_contact_matrices_key_to_index):
    """Initialize the simulation."""

    self.state_mappings = py_state_mappings
    self.storage = {}

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

        self.nodes[ni].linear_coeffs = <DTYPE_t **> malloc(len(pn.linear_coeffs) * sizeof(DTYPE_t*))
        for o in range(len(pn.linear_coeffs)):
            self.nodes[ni].linear_coeffs[o] = <DTYPE_t *> malloc(len(pn.linear_coeffs[o]) * sizeof(DTYPE_t))
            for i in range(len(pn.linear_coeffs[o])):
                self.nodes[ni].linear_coeffs[o][i] = pn.linear_coeffs[o][i]

        self.nodes[ni].infection_coeffs = <DTYPE_t **> malloc(len(pn.infection_coeffs) * sizeof(DTYPE_t*))
        for o in range(len(pn.infection_coeffs)):
            self.nodes[ni].infection_coeffs[o] = <DTYPE_t *> malloc(len(pn.infection_coeffs[o]) * sizeof(DTYPE_t))
            for i in range(len(pn.infection_coeffs[o])):
                self.nodes[ni].infection_coeffs[o][i] = pn.infection_coeffs[o][i]

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

        self.cnodes[ni].linear_coeffs = <DTYPE_t **> malloc(len(pcn.linear_coeffs) * sizeof(DTYPE_t*))
        for o in range(len(pcn.linear_coeffs)):
            self.cnodes[ni].linear_coeffs[o] = <DTYPE_t *> malloc(len(pcn.linear_coeffs[o]) * sizeof(DTYPE_t))
            for i in range(len(pcn.linear_coeffs[o])):
                self.cnodes[ni].linear_coeffs[o][i] = pcn.linear_coeffs[o][i]

        self.cnodes[ni].infection_coeffs = <DTYPE_t **> malloc(len(pcn.infection_coeffs) * sizeof(DTYPE_t*))
        for o in range(len(pcn.infection_coeffs)):
            self.cnodes[ni].infection_coeffs[o] = <DTYPE_t *> malloc(len(pcn.infection_coeffs[o]) * sizeof(DTYPE_t))
            for i in range(len(pcn.infection_coeffs[o])):
                self.cnodes[ni].infection_coeffs[o][i] = pcn.infection_coeffs[o][i]

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

    self.class_infections = <int **> malloc(len(py_class_infections) * sizeof(int *))
    self.class_infections_num = <int *> malloc(len(py_class_infections) * sizeof(int))
    for o in range(len(py_class_infections)):
        self.class_infections[o] = <int *> malloc(len(py_class_infections[o]) * sizeof(int))
        self.class_infections_num[o] = len(py_class_infections[o])
        for i in range(len(py_class_infections[o])):
            self.class_infections[o][i] = py_class_infections[o][i]

    self.linear_terms = <int **> malloc(len(py_linear_terms) * sizeof(int *))
    self.linear_terms_num = <int *> malloc(len(py_linear_terms) * sizeof(int))
    for o in range(len(py_linear_terms)):
        self.linear_terms[o] = <int *> malloc(len(py_linear_terms[o]) * sizeof(int))
        self.linear_terms_num[o] = len(py_linear_terms[o])
        for i in range(len(py_linear_terms[o])):
            self.linear_terms[o][i] = py_linear_terms[o][i]

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
        self.linear_coeffs = None
        self.infection_coeffs = None
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
        self.linear_coeffs = None
        self.infection_coeffs = None
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
