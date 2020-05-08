# Compile and import local pyrossgeo module
import os, sys
sys.path.insert(0,'../')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyrossgeo

import datetime
import time

sim_config_path = '../examples/tutorial1-introduction-to-pyrossgeo/SEAIR_network'

t_start = 0
t_end = 24*60*100

#_, dts = pyrossgeo.utils.get_dt_schedule([
#    (0,  1*60),
#    (7*60,  1),
#    (10*60, 2*60),
#    (17*60, 1),
#    (19*60, 2*60)
#], end_time=24*60)

dt = 1

sim = pyrossgeo.Simulation()

X_state = sim.initialize(sim_config_path=sim_config_path)

start_time = time.time()
dX_state = np.zeros(X_state.size)
sim_data = sim.simulate(X_state, t_start, t_end, dt, steps_per_save=1)
end_time = time.time()
sim_time = (end_time - start_time)/(60*60)
print("Simulation complete. Run-time (h): %s" % sim_time)

ts, node_data, cnode_data, location_data, community_data, network_data = pyrossgeo.utils.get_simulation_data(sim_data)

ts_days = ts / (24*60)
ts_hours = ts / 60