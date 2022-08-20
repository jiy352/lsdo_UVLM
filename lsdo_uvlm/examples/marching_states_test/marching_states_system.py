import matplotlib.pyplot as plt
import openmdao.api as om
from ozone2.api import ODEProblem, Wrap

# from lsdo_uvlm.examples.marching_states_test.ode_outputs import Outputs
import csdl_om
import numpy as np
import csdl


class ODESystemModel(csdl.Model):
    '''
    contains
    1. MeshPreprocessing_comp
    2. SolveMatrix
    3. solve_gamma_b_group
    3. seperate_gamma_b_comp
    4. extract_gamma_w_comp
    '''
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')

    def define(self):
        # rename parameters
        n = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        bd_vortex_shapes = surface_shapes
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)

        ode_surface_shape = [(n, ) + item for item in surface_shapes]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((nt, item[1], 3)) for item in surface_shapes
        ]
        wake_vel_shapes = [(x[0] * x[1], 3) for x in wake_vortex_pts_shapes]

        ode_bd_vortex_shapes = ode_surface_shape

        # ODE system with surface gamma's
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            val = np.zeros((n, nt - 1, ny - 1))
            surface_name = surface_names[i]

            surface_gamma_w_name = surface_name + '_gamma_w'
            surface_dgammaw_dt_name = surface_name + '_dgammaw_dt'
            surface_gamma_b_name = surface_name + '_gamma_b'
            #######################################
            #states 1
            #######################################

            surface_gamma_w = self.declare_variable(surface_gamma_w_name,
                                                    shape=val.shape)
            #para for state 1

            surface_gamma_b = self.declare_variable(surface_gamma_b_name,
                                                    shape=((nx - 1) *
                                                           (ny - 1), ))
            #outputs for state 1
            surface_dgammaw_dt = self.create_output(surface_dgammaw_dt_name,
                                                    shape=(n, nt - 1, ny - 1))

            gamma_b_last = csdl.reshape(surface_gamma_b[(nx - 2) * (ny - 1):],
                                        new_shape=(n, 1, ny - 1))

            surface_dgammaw_dt[:, 0, :] = (gamma_b_last -
                                           surface_gamma_w[:, 0, :]) / delta_t
            surface_dgammaw_dt[:, 1:, :] = (
                surface_gamma_w[:, :(surface_gamma_w.shape[1] - 1), :] -
                surface_gamma_w[:, 1:, :]) / delta_t
            #######################################
            #states 2
            #######################################
            # TODO: fix this comments to eliminate first row
            # t=0       [TE,              TE,                 TE,                TE]
            # t = 1,    [TE,              TE+v_ind(TE,w+bd),  TE,                TE] -> bracket 0-1
            # c11 = TE+v_ind(TE,w+bd)

            # t = 2,    [TE,              TE+v_ind(t=1, bracket 0),  c11+v_ind(t=1, bracket 1),   TE] ->  bracket 0-1-2
            # c21 =  TE+v_ind(t=1, bracket 0)
            # c22 =  c11+v_ind(t=1, bracket 1)

            # t = 3,    [TE,              TE+v_ind(t=2, bracket 0),  c21+vind(t=2, bracket 1), c22+vind(t=2, bracket 2)] -> bracket 0-1-2-3
            # Then, the shedding is

            surface_wake_coords_name = surface_name + '_wake_coords'
            surface_dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            #states 2
            surface_wake_coords = self.create_input(surface_wake_coords_name,
                                                    shape=(n, nt - 1, ny, 3))

            #para's for state 2

            wake_total_vel = self.declare_variable(v_total_wake_names[i],
                                                   val=np.zeros(
                                                       (n, nt - 1, ny, 3)))

            surface = self.declare_variable(surface_name, shape=(n, nx, ny, 3))

            # wake_total_vel_reshaped = csdl.reshape(wake_total_vel,
            #                                        (n, nt, ny, 3))

            surface_dwake_coords_dt = self.create_output(
                surface_dwake_coords_dt_name, shape=((n, nt - 1, ny, 3)))

            TE = surface[:, nx - 1, :, :]

            surface_dwake_coords_dt[:, 0, :, :] = (
                TE - surface_wake_coords[:, 0, :, :] +
                wake_total_vel[:, 0, :, :]) / delta_t

            # print('shapes for the outputs')
            # print('surface_dwake_coords_dt[:, 1:, :, :]',
            #       surface_dwake_coords_dt[:, 1:, :, :].shape)
            # print(
            #     'surface_wake_coords[:, :(surface_wake_coords.shape[1] - 1), :, :]',
            #     surface_wake_coords[:, :(surface_wake_coords.shape[1] -
            #                              1), :, :].shape)
            # print('surface_wake_coords[:, 1:, :, :]',
            #       surface_wake_coords[:, 1:, :, :].shape)
            # print('wake_total_vel[:, 1:, :, :]',
            #       wake_total_vel[:, 1:, :, :].shape)
            surface_dwake_coords_dt[:, 1:, :, :] = (
                surface_wake_coords[:, :
                                    (surface_wake_coords.shape[1] - 1), :, :] -
                surface_wake_coords[:, 1:, :, :] +
                wake_total_vel[:, 1:, :, :]) / delta_t
