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


import time

import openmdao.api as om
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_system_free import ODESystemModel
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_profile_outputs import ProfileSystemModel
from ozone.api import ODEProblem
import csdl

import numpy as np
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show

from lsdo_uvlm.uvlm_preprocessing.actuation_model_temp import ActuationModel
from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh

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

class ODEProblemTest(ODEProblem):
    def setup(self):

        ####################################
        # ode parameter names
        ####################################        
        self.add_parameter('u', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('v', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('w', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('p', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('q', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('r', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('theta',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('x',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('y',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('z',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('psi',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('gamma',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('psiw',dynamic=True, shape=(self.num_times, 1))

        gamma_w_name_list = []
        wing_wake_coords_name_list = []

        for i in range(len(surface_names)):
            # print('surface_names in run.py',surface_names)
            # print('surface_shapes in run.py',surface_shapes)
            ####################################
            # ode parameter names
            ####################################
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            self.add_parameter(surface_name,
                               dynamic=True,
                               shape=(self.num_times, nx, ny, 3))
            ####################################
            # ode states names
            ####################################
            gamma_w_name = surface_name + '_gamma_w'
            wing_wake_coords_name = surface_name + '_wake_coords'
            # gamma_w_name_list.append(gamma_w_name)
            # wing_wake_coords_name_list.append(wing_wake_coords_name)
            # Inputs names correspond to respective upstream CSDL variables
            ####################################
            # ode outputs names
            ####################################
            dgammaw_dt_name = surface_name + '_dgammaw_dt'
            dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            ####################################
            # IC names
            ####################################
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            ####################################
            # states and outputs names
            ####################################
            gamma_w_int_name = surface_name + '_gamma_w_int'
            wake_coords_int_name = surface_name + '_wake_coords_int'
            self.add_state(gamma_w_name,
                           dgammaw_dt_name,
                           shape=(nt - 1, ny - 1),
                           initial_condition_name=gamma_w_0_name,
                           output=gamma_w_int_name)
            self.add_state(wing_wake_coords_name,
                           dwake_coords_dt_name,
                           shape=(nt - 1, ny, 3),
                           initial_condition_name=wake_coords_0_name,
                           output=wake_coords_int_name)

            ####################################
            # profile outputs
            ####################################
            # F_name = surface_name + '_F'
            # self.add_profile_output(F_name)
        

        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.set_ode_system(ODESystemModel)


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_times')
        self.parameters.declare('h_stepsize')

    def define(self):
        num_times = self.parameters['num_times']

        h_stepsize = self.parameters['h_stepsize']

        ####################################
        # Create parameters
        ####################################
        for data in AcStates_val_dict:
            string_name = data
            val = AcStates_val_dict[data]            
            # print('{:15} = {},shape{}'.format(string_name, val, val.shape))

            variable = self.create_input(string_name,
                                         val=val)

        initial_mesh_names = [
            x + '_initial_mesh' for x in surface_names
        ]

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            ####################################
            # Create parameters
            ####################################
            '''1. wing'''
            wing_val = mesh_val
            wing = self.create_input(surface_name, wing_val)

            ########################################
            # Initial condition for states
            ########################################
            '''1. wing_gamma_w_0'''
            wing_gamma_w_0 = self.create_input(gamma_w_0_name, np.zeros((nt - 1, ny - 1)))

            '''2. wing_wake_coords_0'''
            wing_wake_coords_0_val = np.zeros((nt - 1, ny, 3))
            wing_wake_coords_0 = self.create_input(wake_coords_0_name, wing_wake_coords_0_val)

        ########################################
        # Timestep vector
        ########################################
        h_vec = np.ones(num_times - 1) * h_stepsize
        h = self.create_input('h', h_vec)
        ########################################
        # params_dict to the init of ODESystem
        ########################################
        params_dict = {
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': delta_t,
            'nt': nt,
        }

        profile_params_dict = {
            'num_nodes': nt-1,
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': delta_t,
            'nt': nt,
        }

        # add an actuation model on the upstream
        # self.add(ActuationModel(surface_names=surface_names, surface_shapes=surface_shapes, num_nodes=nt-1),'actuation_temp')

        # Create Model containing integrator
        # ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching', num_times, display='default', visualization='None')
        ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching checkpointing', num_times, display='default', visualization='None')

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict), 'subgroup')
        self.add(ProfileSystemModel(**profile_params_dict),'profile_outputs')
        self.add_design_variable('u',lower=1e-3, upper=10)
        # self.add_objective('res')


if __name__ == "__main__":
    # Script to create optimization problem
    be = 'python_csdl_backend'
    make_video = 1


    # define the problem
    num_nodes = 3
    nt = num_nodes+1

    alpha = np.deg2rad(0)

    # define the direction of the flapping motion (hardcoding for now)

    # u_val = np.concatenate((np.array([0.01, 0.5,1.]),np.ones(num_nodes-3))).reshape(num_nodes,1)
    u_val = np.ones((num_nodes, 1))*0.4

    AcStates_val_dict = {
        'u': u_val,
        'v': np.zeros((num_nodes, 1)),
        'w': np.zeros((num_nodes, 1)),
        'p': np.zeros((num_nodes, 1)),
        'q': np.zeros((num_nodes, 1)),
        'r': np.zeros((num_nodes, 1)),
        'theta': np.ones((num_nodes, 1))*alpha,
        'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)),
        'y': np.zeros((num_nodes, 1)),
        'z': np.zeros((num_nodes, 1)),
        'phiw': np.zeros((num_nodes, 1)),
        'gamma': np.zeros((num_nodes, 1)),
        'psiw': np.zeros((num_nodes, 1)),
    }
    ########################################
    # define mesh here
    ########################################
    num_pts_L = 30
    num_pts_R = ny = 23
    nx = num_pts_L-10

    L = 1.
    s_1_ind = 5
    s_2_ind = 26
    num_fish = 1
    grid,h = generate_eel_carling_vlm(num_pts_L,num_pts_R,L,s_1_ind,s_2_ind)

    mesh = generate_simple_mesh_mesh(num_pts_L,num_pts_R)

    x= mesh[:, :, 0]
    y= mesh[:, :, 1]
    z= mesh[:, :, 2]*np.einsum('i,j->ji',np.ones(num_pts_R),h)

    grid = pv.StructuredGrid(x,y,z)
    grid.plot(show_edges=True, show_bounds=False, show_grid=False, show_axes=True)
    # grid.save('vtk/fish.vtk')

    num_time_steps = num_nodes

    lambda_         = 1
    f               = 0.2
    n_period = 0.5

    t_list          = np.linspace(0, 1/f*n_period, num_time_steps)

    mid_line_points = x[:,5]
    num_time_steps  = t_list.size

    x = (mid_line_points-mid_line_points[0]) / (mid_line_points[-1]-mid_line_points[0])
    L = 1           # L is 1 because we scaled x to [0, 1]

    fig = plt.figure()
    ax = fig.add_subplot()
    legend_list = []
    y = np.zeros((num_time_steps,x.size))
    y_dot = y.copy()
    y_lateral = np.zeros((num_time_steps,mid_line_points.size))
    mesh_val = np.zeros((num_nodes, nx, ny, 3))
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
        # grid.save('vtks_fish/fish'+str(i)+'.vtk')
        # mesh_val[i,:,:,0] = x[-nx:,:]
        # mesh_val[i,:,:,1] = y[-nx:,:]
        # mesh_val[i,:,:,2] = z[-nx:,:]

        mesh_val[i,:,:,0] = mesh[-nx:,:,0]
        mesh_val[i,:,:,1] = mesh[-nx:,:,1]+np.einsum('i,j->ij', y_lateral[i],np.ones(num_pts_R))[-nx:,:]
        mesh_val[i,:,:,2] = mesh[-nx:,:,2]
    plt.show()



    surface_names=['eel']
    surface_shapes=[(nx, ny, 3)]
    h_stepsize = delta_t = t_list[1]-t_list[0]

    import python_csdl_backend
    sim = python_csdl_backend.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')

    t_start = time.time()
    sim.run()
    print('simulation time is', time.time() - t_start)
    ######################################################
    # make video
    ######################################################

    if make_video == 1:

        x = sim['eel'][2, :, :, 0]
        y = sim['eel'][2, :, :, 1]
        z = sim['eel'][2, :, :, 2]

        # Create and structured surface
        grid = pv.StructuredGrid(x, y, z)
        # Create a plotter object and set the scalars to the Z height
        plotter = pv.Plotter(notebook=False, off_screen=True)
        plotter.add_mesh(
            grid,
            # scalars=z.ravel(),
            lighting=False,
            show_edges=True,
            scalar_bar_args={"title": "Height"},
            clim=[-1, 1],
        )
        x_wake = sim["eel_wake_coords_int"][2,0,:,0]
        y_wake = sim["eel_wake_coords_int"][2,0,:,1]
        z_wake = sim["eel_wake_coords_int"][2,0,:,2]

        # plotter.add_mesh(
        #     pv.StructuredGrid(x_wake, y_wake, z_wake),
        #     # scalars=z.ravel(),
        #     lighting=False,
        #     show_edges=True,
        #     scalar_bar_args={"title": "Height"},
        #     clim=[-1, 1],
        # )

        # Open a gif
        plotter.open_gif("eel.gif")
        pts = grid.points.copy()
        wk_pts = pv.StructuredGrid(x_wake, y_wake, z_wake).points.copy()
        # Update Z and write a frame for each updated position
        nframe = num_nodes
        i=2
        for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe-2]:
            x = sim['eel'][i, :, :, 0]
            y = sim['eel'][i, :, :, 1]
            z = sim['eel'][i, :, :, 2]
            x_wake = sim["eel_wake_coords_int"][i,:i,:,0]
            y_wake = sim["eel_wake_coords_int"][i,:i,:,1]
            z_wake = sim["eel_wake_coords_int"][i,:i,:,2]
            print("x_wake shape",x_wake.shape)
            grid = pv.StructuredGrid(x, y, z)
            grid_wake = pv.StructuredGrid(x_wake, y_wake, z_wake)
            # grid_wake.plot(show_edges=True, show_bounds=False, show_grid=False, show_axes=True)
            plotter.update_coordinates(grid.points.copy(), render=False)
            # plotter.update_coordinates(grid_wake.points.copy(), render=False)
            # plotter.update_scalars(z.ravel(), render=False)
            # Write a frame. This triggers a render.
            plotter.write_frame()
            i+=1
        # Closes and finalizes movie
        plotter.close()