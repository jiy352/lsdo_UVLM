import time

import matplotlib.pyplot as plt
import openmdao.api as om
from lsdo_uvlm.examples.marching_states_test.marching_states_system import ODESystemModel
from ozone.api import ODEProblem
import csdl
import csdl_om
import csdl_lite
import numpy as np
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show

t_start = time.time()


def generate_simple_mesh(nx, ny, nt=None, offset=0):
    if nt == None:
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0. + offset
    else:
        mesh = np.zeros((nt, nx, ny, 3))
        for i in range(nt):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            if i == 0:
                mesh[i, :, :, 2] = 0.
            else:
                mesh[i, :, :, 2] = mesh[i - 1, :, :, 2] + offset
    return mesh


class ODEProblemTest(ODEProblem):
    def setup(self):
        # # Define field outputs, profile outputs, states, parameters, times
        # # Outputs. coefficients for field outputs must be defined as an upstream variable
        # self.add_field_output('field_output',
        #                       state_name='wing_gamma_w',
        #                       coefficients_name='coefficients')

        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        for i in range(len(surface_names)):
            ####################################
            # ode parameter names
            ####################################
            surface_name = surface_names[i]
            gamma_b_name = surface_name + '_gamma_b'
            wake_total_vel_name = surface_name + '_wake_total_vel'

            self.add_parameter(gamma_b_name,
                               dynamic=True,
                               shape=(self.num_times, (nx - 1) * (ny - 1)))
            self.add_parameter(surface_name,
                               dynamic=True,
                               shape=(self.num_times, nx, ny, 3))
            self.add_parameter(wake_total_vel_name,
                               dynamic=True,
                               shape=(self.num_times, nt - 1, ny, 3))

            ####################################
            # ode states names
            ####################################
            gamma_w_name = surface_name + '_gamma_w'
            wing_wake_coords_name = surface_name + '_wake_coords'
            # # Inputs names correspond to respective upstream CSDL variables
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
            # profile outputs names
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

        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.set_ode_system(ODESystemModel)


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_times')

    def define(self):
        num_times = self.parameters['num_times']

        h_stepsize = 1

        ####################################
        # Create parameter for parameters
        ####################################
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            gamma_b_name = surface_name + '_gamma_b'
            wake_total_vel_name = surface_name + '_wake_total_vel'
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            '''1. gamma_b'''
            gamma_b_val = np.zeros(
                (num_times, (nx - 1) *
                 (ny - 1)))  # dynamic parameter defined at every timestep
            for t in range(num_times):
                gamma_b_val[
                    t] = 1.0 + t / num_times / 5.0  # dynamic parameter defined at every timestep
                # print(gamma_b_val)
            gamma_b = self.create_input(gamma_b_name, gamma_b_val)
            '''2. wing'''
            wing_val = generate_simple_mesh(nx, ny, num_times, offset=0.2)
            wing = self.create_input(surface_name, wing_val)
            '''3. wing_wake_total_vel'''
            wing_wake_total_vel_val = np.zeros((num_times, nt - 1, ny, 3))
            wing_wake_total_vel_val[0, 0, :, 0] = 1
            for t in range(num_times):
                # wing_wake_total_vel_val[t, 0:t + 1, :, 0] = t + 1
                wing_wake_total_vel_val[t, 0:t + 1, :, 0] = 1
            wing_wake_total_vel = self.create_input(wake_total_vel_name,
                                                    wing_wake_total_vel_val)

            ########################################
            # Initial condition for states
            ########################################
            '''1. wing_gamma_w_0'''
            wing_gamma_w_0 = self.create_input(gamma_w_0_name,
                                               np.zeros((nt - 1, ny - 1)))

            # TODO: need to test if this works
            '''2. wing_wake_coords_0'''
            wing_wake_coords_0_val = np.zeros((nt - 1, ny, 3))
            wing_wake_coords_0 = self.create_input(wake_coords_0_name,
                                                   wing_wake_coords_0_val)
        # wing_wake_coords_0 = self.create_output('wing_wake_coords_0',
        #                                         shape=(nt, ny, 3))
        # TE_0 = wing[0, nx - 1, :, :]
        # wing_wake_coords_0 = csdl.expand(wing[0, nx - 1, :, :].reshape(ny, 3),
        #                                  (nt, ny, 3), 'ij->kij')
        # wing(num_times, nx,ny,3), TE(1, 1,ny,3)

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

        # Create Model containing integrator
        ODEProblem = ODEProblemTest('ForwardEuler',
                                    'time-marching',
                                    num_times,
                                    display='default',
                                    visualization='None')
        # ODEProblem = ODEProblemTest('RK4', 'solver-based', num_times, display='default', visualization='None')

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict),
                 'subgroup', ['*'])


if __name__ == "__main__":
    # Script to create optimization problem
    # be = 'csdl_om'
    be = 'csdl_lite'
    make_video = 1

    nx = 4
    ny = 10
    h_stepsize = delta_t = 1.

    h_initial = 1
    num_nodes = 1
    surface_names = ['wing']
    surface_shapes = [(nx, ny)]
    delta_t = 1.
    nt = 5
    if be == 'csdl_om':
        sim = csdl_om.Simulator(RunModel(num_times=nt - 1), mode='rev')
    if be == 'csdl_lite':
        sim = csdl_lite.Simulator(RunModel(num_times=nt - 1), mode='rev')
    sim.run()
    # sim.check_partials(compact_print=True)
    print('simulation time is', time.time() - t_start)
    print('#' * 50, 'print states', '#' * 50)
    print('wing_gamma_w_int', sim['wing_gamma_w_int'])
    print('wing_wake_coords_int', sim['wing_wake_coords_int'])

    print('#' * 50, 'print wings', '#' * 50)
    print('wing', sim['wing'])

    print('#' * 50, 'print wing_wake_total_vel', '#' * 50)
    print('wing_wake_total_vel', sim['wing_wake_total_vel'])

    if make_video == 1:

        axs = Axes(
            xrange=(0, 20),
            yrange=(-10, 10),
            zrange=(-0.2, 3),
        )

        video = Video("spider.gif", duration=2, backend='ffmpeg')
        for i in range(nt - 1):
            vp = Plotter(
                bg='beige',
                bg2='lb',
                # axes=0,
                #  pos=(0, 0),
                offscreen=False,
                interactive=1)
            # Any rendering loop goes here, e.g.:
            vps = Points(np.reshape(sim['wing'][i, :, :, :], (-1, 3)),
                         r=8,
                         c='red')
            vp += vps
            vp += __doc__
            vps = Points(np.reshape(sim['wing_wake_coords_int'][i, 0:i, :, :],
                                    (-1, 3)),
                         r=8,
                         c='blue')
            vp += vps
            vp += __doc__
            # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
            # video.action(cameras=[cam1, cam1])

            vp.show(axs, elevation=-75, azimuth=0,
                    axes=False)  # render the scene

            video.addFrame()  # add individual frame

        vp.closeWindow()

        video.close()  # merge all the recorded frames

        vp.interactive().close()