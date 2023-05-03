import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from panel_method.utils.generate_eel_vlm_mesh import (
    generate_eel_carling_vlm,get_connectivity_matrix_from_pyvista,neighbor_cell_idx, get_swimming_eel_geometry)
# from panel_method.utils.generate_meshes import get_connectivity_matrix_from_pyvista

from panel_method.mesh_preprocessing.generate_kinematics import eel_kinematics
from panel_method.utils.generate_meshes import generate_fish_v_m1#,get_connectivity_matrix_from_pyvista
from panel_method.upm_solver_dynamic_ode import  UPMSolverDynamicODE
from panel_method.mesh_preprocessing.preprocessing import (find_te_info_NACA, generate_wake_panels)
# from panel_method.utils.generate_meshes import (read_vtk_unstruct,neighbor_cell_idx)

# This code needs three inputs
# ODE_parameters=None, geometric_dict, and kinematics_dict
# This programe is the analysis fish swimming performance with the prescribed wake
# init date: Feb 27 2023
# Jiayao Yan

##########################################
# 1. Define inputs for the solver        #
##########################################
v_inf = 0.4 # body length per second
filename = 'vtk/fish.vtk'
alpha=0

# 1.1 define the fish geometry
##########################################

num_time_steps=26

num_pts_L = 50
num_pts_R = 23
L = 1.
s_1_ind = 5
s_2_ind = 45
num_fish = 1

grid,h = generate_eel_carling_vlm(num_pts_L,num_pts_R,L,s_1_ind,s_2_ind)


def generate_simple_mesh_mesh(nx, ny, nt=None):
    if nt == None:
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))/(nx-1)
        mesh[:, :, 1] = 0.
        mesh[:, :, 2] = np.outer(np.arange(ny), np.ones(nx)).T/(ny-1)-0.5
    else:
        mesh = np.zeros((nt, nx, ny, 3))
        for i in range(nt):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[i, :, :, 2] = 0.
    return mesh

import pyvista as pv
import numpy as np

mesh = generate_simple_mesh_mesh(num_pts_L,num_pts_R)
x= mesh[:, :, 0]
y= mesh[:, :, 1]
z= mesh[:, :, 2]*np.einsum('i,j->ji',np.ones(num_pts_R),h)

grid = pv.StructuredGrid(x,y,z)
grid.plot(show_edges=True, show_bounds=False, show_grid=False, show_axes=True)


grid.save('vtk/fish.vtk')

lambda_         = 1
f               = 0.2

t_list          = np.linspace(0, 1/f, num_time_steps)

mid_line_points = x[:,5]
num_time_steps  = t_list.size

# rescaling x from 0 to 1 assuming x is monotonically increasing
x = (mid_line_points-mid_line_points[0]) / (mid_line_points[-1]-mid_line_points[0])
L = 1           # L is 1 because we scaled x to [0, 1]

fig = plt.figure()
ax = fig.add_subplot()
legend_list = []
y = np.zeros((num_time_steps,x.size))
y_dot = y.copy()
y_lateral = np.zeros((num_time_steps,mid_line_points.size))
for i in range(num_time_steps):
    t = t_list[i]
    omg = 2*np.pi*f
    y_lateral[i,:] = 0.125*((mid_line_points+0.03125)/(1.03125))*np.sin(np.pi*2*mid_line_points/lambda_ - omg*t)
    y_lateral[i,:] =  0.125*((mid_line_points+0.03125)/(1.03125))*np.cos(np.pi*2*mid_line_points/lambda_ - omg*t)*(-omg)
    # plt.plot(x,y_dot[i],'.')
    # plt.plot(x,y[i])
    x= mesh[:, :, 0]
    y= mesh[:, :, 1]+np.einsum('i,j->ij', y_lateral[i],np.ones(num_pts_R))
    z= mesh[:, :, 2]*np.einsum('i,j->ji',np.ones(num_pts_R),h)
    grid= pv.StructuredGrid(x,y,z)
    grid.save('vtks_fish/fish'+str(i)+'.vtk')

plt.show()