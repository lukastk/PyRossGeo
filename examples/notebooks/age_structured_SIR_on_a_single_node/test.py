# Compile and import local geoSIR module
import os, sys

sys.path.insert(0,'..')
sys.path.insert(0,'../..')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sim_utils import *
import geodemic

import datetime
import time

t_start = 0
t_end = 24*60*800
dt = 1

_, dts = get_dt_schedule([
    (0,  1*60),
    (7*60,  1),
    (10*60, 2*60),
    (17*60, 1),
    (19*60, 2*60)
], end_time=24*60)

#_, dts = get_dt_schedule([
#    (0,  1)
#], end_time=24*60)

model_path = 'model.json' 
commuter_networks_path = 'commuter_networks.csv'
node_parameters_path = 'node_parameters.csv'
cnode_parameters_path = 'cnode_parameters.csv' 
contact_matrices_path = 'contact_matrices.json' 
node_cmatrices_path = 'node_cmatrices.csv' 
cnode_cmatrices_path = 'cnode_cmatrices.csv' 
node_positions_path = 'node_positions.csv' 
node_populations_path = 'node_populations.csv' 

sim = geodemic.simulation()

X_state = sim.initialize(model_path, commuter_networks_path,
                            node_parameters_path, cnode_parameters_path,
                            contact_matrices_path, node_cmatrices_path, cnode_cmatrices_path,
                            node_populations_path)

start_time = time.time()
dX_state = np.zeros(X_state.size)
sim_data = sim.simulate(X_state, t_start, t_end, dts, steps_per_save=len(dts))
end_time = time.time()
sim_time = (end_time - start_time)/(60*60)
print("Simulation complete. Run-time (h): %s" % sim_time)

ts, node_data, cnode_data, location_data, community_data, network_data = geodemic.utils.get_simulation_data(sim_data)

ts = ts / (24*60)