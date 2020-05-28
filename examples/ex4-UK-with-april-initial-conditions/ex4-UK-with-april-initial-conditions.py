# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %%capture
# Compile and import local pyrossgeo module
import os, sys
owd = os.getcwd()
os.chdir('../../')
sys.path.insert(0,'../../')
# !python setup.py build_ext --inplace
os.chdir(owd)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyrossgeo

import pandas as pd
import json
# -

# We will generate most of the simulation configuration files based on parameters we feed into the simulation.
#
# Warnings about RAM
#
# Allow to have names in the node populations
#
# Explanation about how node pop and commtuer network awas construced.
#
# Explain what age groups we have
#
# Mention that we set `min_num_moving` quite high here for demonstration purposes.

# [00:13, 27/05/2020] Jakub: betaEffec:
#  array([1.34733454, 3.36833634, 4.71567087, 5.38933814, 6.73667268,
#         6.73667268, 6.73667268]
#         
# [00:13, 27/05/2020] Jakub: For age_groups = [16, 25, 35, 50, 65, 75]
#
# [00:14, 27/05/2020] Jakub: For Lad 5 in my list
#
# [00:14, 27/05/2020] Jakub: sus= np.array([0.2, 0.5, 0.7, 0.8, 1,  1, 1])
#
# [00:15, 27/05/2020] Jakub: (this is the susceptibility to infection upon contact with infected)
#
# [00:15, 27/05/2020] Jakub: beta=betaEffec*sus
#
# [00:15, 27/05/2020] Jakub: Shouldd I put this in some document?
#
# [00:15, 27/05/2020] Jakub: There is quite a lot of it

# [01:06, 27/05/2020] Jakub: z=[3,2,2,3,3,2,1]
#
# ```
# i1, j1 =0, 0
# for l in range(M):
#     j1=0
#     for j in range(M):
#         CH[l,j] = np.sum( CH0[i1:i1+z[l], j1:j1+z[l]] )
#         CW[l,j] = np.sum( CW0[i1:i1+z[l], j1:j1+z[l]] )
#         CS[l,j] = np.sum( CS0[i1:i1+z[l], j1:j1+z[l]] )
#         CO[l,j] = np.sum( CO0[i1:i1+z[l], j1:j1+z[l]] )
#         j1 = j1+z[j]
#     i1 = i1+z[l]
# ```    
#     
# [01:07, 27/05/2020] Jakub: Although you might want to split the first one
# I have 9 age groups right?
# z=[1,1,1,2,2,3,3,2,1]

# For your model alpha, betaEffective (contactmatrix as well)

# Plan:
#
# - Jakub has to redo the inference but with adjusted contact matrices.
# - Use g^\ell instead of g^\ell_{ij}.
# - Rescale the contact matrices temporally, but don't try to adjust them based on how many people are working.
# - Perhaps implement the rescaling formula directly into pyrossgeo.

# # Generate the configuration files

# ### Configuration generation parameters
#
# Here we define some parameters with which all the configuration files will be generated. Edit these if you want to change the simulation.

# Age groups:
# 0-4, 5-11, 12-15, 16-24, 25-34, 35-49, 50-64, 65-74

# +
sim_config_path = 'uk_march_20'

min_num_moving = 500 # Remove all commuting edges where less than `min_num_moving` are moving

# Decide which classes are allowed to commute
allow_class = [
    ('S', True),
    ('E', True),
    ('A', True),
    ('Ia1', True),
    ('Ia2', True),
    ('Ia3', True),
    ('Is1', False),
    ('Is2', False),
    ('Is3', False),
    ('R', True),
]

age_groups = 9

# Node parameters

n_gammaE = 1/2.72
n_gammaA = 1/3.12
n_gammaIa = 1/7.0
n_gammaIs = 1/4.82

n_gammaIs1 = n_gammaIs2 = n_gammaIs3 = n_gammaIs / 3
n_gammaIa1 = n_gammaIa2 = n_gammaIa3 = n_gammaIa / 3

self_isolation = 0.1
n_beta = 0.51695631

alpha = np.array([0.45, 0.56258112, 0.49953083, 0.44354679, 0.36383239, 0.28402315, 0.21310739, 0.16472101, 0.13511729])
sus = np.array([0.2, 0.2, 0.2, 0.5, 0.7, 0.8, 1, 1, 1])

# Cnode parameters

cn_gammaE = n_gammaE
cn_gammaA = n_gammaA
cn_gammaIa = n_gammaIa
cn_gammaIs = n_gammaIs

cn_gammaIs1 = cn_gammaIs2 = cn_gammaIs3 = n_gammaIs1
cn_gammaIa1 = cn_gammaIa2 = cn_gammaIa3 = n_gammaIa1

self_isolation = 1
cn_beta = n_beta
cn_betaA = cn_beta
cn_betaIa1 = cn_betaIa2 = cn_betaIa3 = cn_beta
cn_betaIs1 = cn_betaIs2 = cn_betaIs3 = cn_beta*self_isolation

# Time steps

t_start = 0
t_end = 24*60*100

_, dts = pyrossgeo.utils.get_dt_schedule([
    (0,  1*60),
    (7*60,  2),
    (10*60, 2*60),
    (17*60, 2),
    (19*60, 2*60)
], end_time=24*60)
# -

# ### Model

# +
import json

with open('%s/model.json' % sim_config_path, 'r') as f:
    model = json.load(f)   
with open('%s/model.json' % sim_config_path, 'r') as f:
    print(f.read())
# -

# ### Format the commuting network

# +
cn = pd.read_csv("%s/commuter_networks.csv" % sim_config_path)

#### Set which classes are allowed to commute

# Drop the current allow_O columns
cn = cn.iloc[:,:10]

# Set allow settings
for O, allow_O in allow_class:
    cn[ "Allow %s" % O ] = 1 if allow_O else 0
    
# Allow people to return home
cn.loc[ cn['Home'] == cn['To'],"Allow %s" % allow_class[0][0]:] = 1

#### Remove commuting edges where fewer than `min_num_moving` people are commuting

delete_rows = []

for i, row in cn.loc[ cn['Home'] == cn['From'] ].iterrows():
    if row['# to move'] < min_num_moving:
        delete_rows.append(i)
        delete_rows.append(i+1) # Delete the returning commuting edge as well

cn = cn.reset_index()
cn = cn.drop(delete_rows)
cn = cn.drop(columns='index')

cn.loc[cn['ct1'] == cn['ct2'], 'ct2'] += 0.1

cn.head()
# -

# ### Setting the node and cnode parameters

# We need to add rows giving the model parameters in `node_parameters.csv` and `cnode_parameters.csv`, which currently only has the areas of each geographical node:

# +
nparam = pd.read_csv('%s/node_parameters.csv' % sim_config_path)

nparam = nparam.append({
    'Home' : 'ALL',
    'Location' : 'ALL',
    'Age' : 'ALL',
    'gammaE' : n_gammaE,
    'gammaIa1' : n_gammaIa1,
    'gammaIa2' : n_gammaIa2,
    'gammaIa3' : n_gammaIa3,
    'gammaIs1' : n_gammaIs1,
    'gammaIs2' : n_gammaIs2,
    'gammaIs3' : n_gammaIs3
}, ignore_index=True)

# Age dependent rates

for age in range(age_groups):
    b = n_beta*sus[age]
    n_betaA = b
    n_betaIa1 = n_betaIa2 = n_betaIa3 = b
    n_betaIs1 = n_betaIs2 = n_betaIs3 = b*self_isolation
    
    nparam = nparam.append({
        'Home' : 'ALL',
        'Location' : 'ALL',
        'Age' : age,
        'betaA' : n_betaA,
        'betaIa1' : n_betaIa1,
        'betaIa2' : n_betaIa2,
        'betaIa3' : n_betaIa3,
        'betaIs1' : n_betaIs1,
        'betaIs2' : n_betaIs2,
        'betaIs3' : n_betaIs3,
        'alpha*gammaA' : n_gammaA*alpha[age],
        'alphabar*gammaA' : n_gammaA*(1-alpha[age]),
    }, ignore_index=True)

nparam.iloc[-10:,:]

# +
cnparam = pd.read_csv('%s/cnode_parameters.csv' % sim_config_path)

cnparam = cnparam.append({
    'Home' : 'ALL',
    'From' : 'ALL',
    'To' : 'ALL',
    'Age' : 'ALL',
    'gammaE' : n_gammaE,
    'gammaIa1' : n_gammaIa1,
    'gammaIa2' : n_gammaIa2,
    'gammaIa3' : n_gammaIa3,
    'gammaIs1' : n_gammaIs1,
    'gammaIs2' : n_gammaIs2,
    'gammaIs3' : n_gammaIs3
}, ignore_index=True)

# Age dependent rates

for age in range(age_groups):
    b = n_beta*sus[age]
    n_betaA = b
    n_betaIa1 = n_betaIa2 = n_betaIa3 = b
    n_betaIs1 = n_betaIs2 = n_betaIs3 = b*self_isolation
    
    cnparam = cnparam.append({
        'Home' : 'ALL',
        'From' : 'ALL',
        'To' : 'ALL',
        'Age' : age,
        'betaA' : n_betaA,
        'betaIa1' : n_betaIa1,
        'betaIa2' : n_betaIa2,
        'betaIa3' : n_betaIa3,
        'betaIs1' : n_betaIs1,
        'betaIs2' : n_betaIs2,
        'betaIs3' : n_betaIs3,
        'alpha*gammaA' : n_gammaA*alpha[age],
        'alphabar*gammaA' : n_gammaA*(1-alpha[age]),
    }, ignore_index=True)

cnparam
# -

# ### Contact matrices
#
# Define the contact matrices

# +
CH0, CW0, CS0, CO0 = np.load('%s/contact_matrices.npy' % sim_config_path)

C_home = np.zeros((age_groups, age_groups))
C_work = np.zeros((age_groups, age_groups))
C_school = np.zeros((age_groups, age_groups))
C_other = np.zeros((age_groups, age_groups))
z=[1,1,1,2,2,3,3,2,1]
i1, j1 =0, 0
for l in range(age_groups):
    j1=0
    for j in range(age_groups):
        C_home[l,j] = np.sum( CH0[i1:i1+z[l], j1:j1+z[l]] )
        C_work[l,j] = np.sum( CW0[i1:i1+z[l], j1:j1+z[l]] )
        C_school[l,j] = np.sum( CS0[i1:i1+z[l], j1:j1+z[l]] )
        C_other[l,j] = np.sum( CO0[i1:i1+z[l], j1:j1+z[l]] )
        j1 = j1+z[j]
    i1 = i1+z[l] 

# Rescale contact matrices from units of days to minutes
    
C_home = C_home / (24*60)
work_hours = 17-9
C_work = C_work / (work_hours*60)
C_school = C_school / (work_hours*60)
C_other = C_other / (24*60)
    
C_no_work = C_home + C_other
C_at_work = C_home + C_work + C_school + C_other
C_commute = C_home + C_work + C_school + C_other

contact_matrices = {
    'C' : C_no_work,
    'C_commute' : C_commute
}

# +
ncm = pd.DataFrame(columns=['Home', 'Location'] + model['settings']['classes'])

ncm = ncm.append({
    'Home' : 'ALL',
    'Location' : 'ALL',
    'A' : 'C',
    'Ia1' : 'C',
    'Ia2' : 'C',
    'Ia3' : 'C',
    'Is1' : 'C',
    'Is2' : 'C',
    'Is3' : 'C'
}, ignore_index=True)

# +
cncm = pd.DataFrame(columns=['Home', 'From', 'To'] + model['settings']['classes'])

cncm = cncm.append({
    'Home' : 'ALL',
    'From' : 'ALL',
    'To' : 'ALL',
    'A' : 'C_commute',
    'Ia1' : 'C_commute',
    'Ia2' : 'C_commute',
    'Ia3' : 'C_commute',
    'Is1' : 'C_commute',
    'Is2' : 'C_commute',
    'Is3' : 'C_commute'
}, ignore_index=True)

# +
minutes_in_day = 24*60*60

def set_work_contact_matrix(sim, step_i, t, dt, X_state, dX_state, X_state_saved, ts_saved, save_i): 
    global C_at_work
    
    tday = t % minutes_in_day
    
    cmat = sim.get_contact_matrix('C')
    cmat[:] = C_at_work
    
def set_no_work_contact_matrix(sim, step_i, t, dt, X_state, dX_state, X_state_saved, ts_saved, save_i): 
    global C_no_work
    
    tday = t % minutes_in_day
    
    cmat = sim.get_contact_matrix('C')
    cmat[:] = C_no_work


# -

# ## Run simulation

# +
sim = pyrossgeo.Simulation()

X_state = sim.initialize(
    model_dat = '%s/model.json' % sim_config_path,
    commuter_networks_dat = cn,
    node_populations_dat = '%s/node_populations.csv' % sim_config_path,
    node_parameters_dat = nparam,
    cnode_parameters_dat = cnparam,
    contact_matrices_dat = contact_matrices,
    node_cmatrices_dat = ncm,
    cnode_cmatrices_dat = cncm
)

sim.add_event([9], set_work_contact_matrix, repeat_time=24*60)
sim.add_event([17], set_no_work_contact_matrix, repeat_time=24*60)

sim_data = sim.simulate(X_state, t_start, t_end, dts, steps_per_save=len(dts), steps_per_print=len(dts), only_save_nodes=True, save_path='out')

ts_days = ts / (24*60)
ts_hours = ts / 60
# -

# ## Plot the result

# Plot the evolution of the whole network

# +
plt.figure( figsize=(8,3) )

S = np.sum(network_data[:,:,0], axis=1)
E = np.sum(network_data[:,:,1], axis=1)
A = np.sum(network_data[:,:,2], axis=1)
I = np.sum(network_data[:,:,3], axis=1)
R = np.sum(network_data[:,:,4], axis=1)

plt.plot(ts_days, S, label="S")
plt.plot(ts_days, E, label="I")
plt.plot(ts_days, A, label="I")
plt.plot(ts_days, I, label="I")
plt.plot(ts_days, R, label="R")

plt.legend(loc='upper right', fontsize=12)
plt.xlabel('Days')
# -

# ### Plotting the result using GeoPandas

# Assemble geo data and define helper functions. Edit `plot_frame` to change the format of the video.

# +
import pickle
import tempfile
import geopandas as gpd
from geopandas.plotting import plot_polygon_collection
from matplotlib import animation

# Simulation data

location_data = pyrossgeo.utils.extract_location_data(sim_data)
N_ = np.sum(location_data[:,:,:,:], axis=(1,2))

S_ = np.sum(location_data[:,:,0,:], axis=1)
E_ = np.sum(location_data[:,:,1,:], axis=1)
A_ = np.sum(location_data[:,:,2,:], axis=1)
I_ = np.sum(location_data[:,:,3,:], axis=1)
R_ = np.sum(location_data[:,:,4,:], axis=1)

s_ = S_ / N_
e_ = E_ / N_
a_ = A_ / N_
i_ = I_ / N_
r_ = R_ / N_

ts_days = pyrossgeo.utils.extract_ts(sim_data) / (24*60)

epi_data = np.sum(np.array([   # Used to plot pandemic curves
    S_,E_,A_,I_,R_
]), axis=2)

# Load geometry

geometry_node_key = 'msoa11cd'
geometry = gpd.read_file("../geodata/london_geo/london_msoa_shapes/Middle_Layer_Super_Output_Areas_December_2011_Boundaries_EW_BGC.shp")

loc_table = pd.read_csv('london_simulation/loc_table.csv')
loc_table_loc_col = loc_table.columns[0]
loc_table_loc_key_col = loc_table.columns[1]

geometry = geometry[ geometry[geometry_node_key].isin(loc_table.iloc[:,1]) ] # Remove locations in geometry that are not in loc_table
geometry = geometry.merge(loc_table, left_on=geometry_node_key, right_on=loc_table_loc_key_col) # Add location indices
geometry = geometry.sort_values(by=loc_table_loc_col) # Sort them by location indices

# Edit this function to adjust the layout of the video

def plot_frame(ti, close_plot=False, tmp_save=None):
    fig, axes = plt.subplots(ncols=3, nrows=2, gridspec_kw={'width_ratios':[1, 1, 1.3]}, figsize=(18, 14))

    geometry['S'] = s_[ti,:]
    geometry['E'] = e_[ti,:]
    geometry['A'] = a_[ti,:]
    geometry['I'] = i_[ti,:]
    geometry['R'] = r_[ti,:]
    
    plot_geo(geometry, axes[0,0], vmin=0, vmax=1, value_key='S', title="Susceptible", legend=False)
    plot_geo(geometry, axes[0,1], vmin=0, vmax=1, value_key='E', title="Exposed", legend=False)
    plot_geo(geometry, axes[0,2], vmin=0, vmax=1, value_key='A', title="Activated", legend=True)
    plot_geo(geometry, axes[1,0], vmin=0, vmax=1, value_key='I', title="Infected", legend=False)
    plot_geo(geometry, axes[1,1], vmin=0, vmax=1, value_key='R', title="Recovered", legend=False)
    
    plot_epi(axes[1,2], ti, ts_days, epi_data, ['S','E','A','I','R'])
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.suptitle("SEAIR Model - Day %s" % ti, fontsize=18)
    
    if not tmp_save is None:
        plt.savefig(tmp_save.name + '/%s.png' % ti)
    if close_plot:
        plt.close(fig)
    if not tmp_save is None:
        return tmp_save.name + '/%s.png' % ti

# Helper functions for plotting

def plot_geo(geometry, ax, vmin, vmax, value_key='val', title="", legend=True, legend_label='', cax=None, axis_on=False):
    if legend:
        if cax is None:
            geometry.plot(column=value_key, ax=ax, vmin=vmin, vmax=vmax, legend=True, legend_kwds={'label': legend_label})
        else:
            geometry.plot(column=value_key, ax=ax, cax=cax, vmin=vmin, vmax=vmax, legend=True, legend_kwds={'label': legend_label})
    else:
        geometry.plot(column=value_key, ax=ax, cax=cax, vmin=vmin, vmax=vmax, legend=False)
        
    ax.set_title(title)
    if not axis_on:
        ax.set_axis_off()
        
def plot_epi(ax, ti, ts, epi_data, epi_data_labels):
    for oi in range(epi_data.shape[0]):
        ax.plot(ts[:ti], epi_data[oi,:ti], label=epi_data_labels[oi])
    ax.legend(loc='center left')
    
    ax.set_xlim(np.min(ts_days), np.max(ts_days))
    ax.set_ylim(0, np.max(epi_data))


# -

# Plot the pandemic at a given day

# +
day = 50

geometry['S'] = s_[day,:]
geometry['E'] = e_[day,:]
geometry['A'] = a_[day,:]
geometry['I'] = i_[day,:]
geometry['R'] = r_[day,:]

fig, ax = plt.subplots(figsize=(7, 5))

plot_geo(geometry, ax, vmin=0, vmax=1, value_key='S', title='Susceptibles at day %s' % day)

# +
day = 50

plot_frame(day)
# -

# Create a video of the pandemic

# +
tmp_dir = tempfile.TemporaryDirectory()

frames_paths = []

for ti in range(len(ts)):
    if ti % 1 == 0:
        print("Frame %s of %s" % (ti, len(ts)))
    frame_path = plot_frame(ti, close_plot=True, tmp_save=tmp_dir)
    frames_paths.append(frame_path)
    
import cv2

video_name = 'sim_video.mp4'

frame = cv2.imread(frames_paths[0])
height, width, layers = frame.shape
fps = 6
#codec=cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
codec=cv2.VideoWriter_fourcc(*'DIVX')

video = cv2.VideoWriter(video_name, codec, fps, (width,height))

for frame_path in frames_paths:
    video.write(cv2.imread(frame_path))

cv2.destroyAllWindows()
video.release()
