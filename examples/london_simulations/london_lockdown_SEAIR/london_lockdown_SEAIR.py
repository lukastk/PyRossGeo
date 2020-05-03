import sys, os
sys.path.insert(0,'../../../')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyrossgeo
import zarr
from shutil import copyfile

import datetime
import time

# Simulation settings

sim_days = 120
t_start = 0
t_end = 24*60*sim_days

_, dts = pyrossgeo.utils.get_dt_schedule([
    (0,  2*60),
    (8*60,  1),
    (10*60, 2*60),
    (17*60, 1),
    (19*60, 2*60)
], end_time=24*60)

steps_per_save = 1

sim_title = str(input("Enter title of simulation: ")).replace(" ", "_")
sim_comments = str(input("Comments about simulation: "))
date = datetime.datetime.now().strftime("%y-%m-%d:%H:%M:%S")

if sim_title == "":
    sim_title = os.path.split( os.getcwd() )[-1]

out_path = '../simulation_results/%s_%s' % (date, sim_title)
out_file = '%s/X_states' % out_path

# System parameters

model_path = 'model.json' 
commuter_networks_path = 'commuter_networks.csv'
node_parameters_path = 'node_parameters.csv'
cnode_parameters_path = 'cnode_parameters.csv' 
contact_matrices_path = 'contact_matrices.json' 
node_cmatrices_path = 'node_cmatrices.csv' 
cnode_cmatrices_path = 'cnode_cmatrices.csv' 
node_positions_path = 'node_positions.csv' 
node_populations_path = 'node_populations.csv'

# Copy system parameters to results folder

system_param_files = [
    model_path,
    commuter_networks_path,
    node_parameters_path,
    cnode_parameters_path,
    contact_matrices_path,
    node_cmatrices_path,
    cnode_cmatrices_path,
    node_positions_path,
    node_populations_path
]

os.makedirs('%s/system_parameters/' % out_path, exist_ok=True)

for pfile in system_param_files:
    copyfile(pfile, '%s/system_parameters/%s' % (out_path, pfile))

# Lockdown

event_times = [ 24*60*30]

def event_function(cg, step_i, t, dt, X_state, dX_state):
    print("Lockdown starting. Day: %s" % (t/(24*60)))
    cmat = cg.get_contact_matrix('C_all')
    cmat_home = cg.get_contact_matrix('C_home')
    cmat[:] = cmat_home
    cg.stop_commuting(True)
    
# Run simulation

sim = pyrossgeo.simulation()
print("Initializing simulation...")
X_state = sim.initialize(model_path, commuter_networks_path,
                            node_parameters_path, cnode_parameters_path,
                            contact_matrices_path, node_cmatrices_path, cnode_cmatrices_path,
                            node_populations_path)
print("Initialization complete.")

print("Simulation starting...")
start_time = time.time()
dX_state = np.zeros(X_state.size)
sim_data = sim.simulate(X_state, t_start, t_end, dts, steps_per_print=steps_per_save,
                                steps_per_save=steps_per_save, out_file=out_file,
                                event_times=event_times, event_function=event_function)
end_time = time.time()
sim_time = (end_time - start_time)/(60*60)

print("Simulation complete. Run-time (h): %s" % sim_time)

### Save results

print("Saving results... Directory: %s" % out_path)

state_mappings, ts_saved, X_states_res = sim_data

# Re-open X_states with more suitables chunks

X_states_res = zarr.open("%s.zarr" % out_file, mode='r', chunks=(X_states_res.shape[0], 1))

# Save state mappings

node_mappings, cnode_mappings = state_mappings

import pickle

with open("%s/node_mappings.pkl" % out_path,"wb") as f:
    pickle.dump(node_mappings, f)

with open("%s/cnode_mappings.pkl" % out_path,"wb") as f:
    pickle.dump(cnode_mappings, f)

# Save simulation protocol, analysis.ipynb and comments

copyfile(__file__, '%s/%s' % (out_path, __file__))

with open("%s/comments.txt" % out_path,"w") as f:
    f.write(sim_comments)

copyfile('Analysis.ipynb', '%s/%s.ipynb' % (out_path, sim_title))

# Save ts

np.save("%s/ts.npy" % out_path, ts_saved)

# Open X_states in RAM

X_states = X_states_res[:]
sim_data = state_mappings, ts_saved, X_states

# Save location_data

print("Attempting to save location_data")

location_data = pyrossgeo.utils.get_location_data(sim_data)
np.save("%s/location_data.npy" % out_path, location_data)
del location_data

#community_data = pyrossgeo.utils.get_community_data(sim_data)
#np.save("%s/community_data.npy" % out_path, community_data)
#del community_data

#network_data = pyrossgeo.utils.get_network_data(sim_data)
#np.save("%s/network_data.npy" % out_path, network_data)
#del network_data

print("Simulation saved.")