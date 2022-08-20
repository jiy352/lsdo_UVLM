import time

import matplotlib.pyplot as plt
import openmdao.api as om
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_free.plunging_system_free import ODESystemModel
# from lsdo_uvlm.examples.simple_wing_constant_aoa_sls.plunging_system_free import ODESystemModel

from ozone.api import ODEProblem
import csdl
import csdl_om
import csdl_lite
import numpy as np
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show


from lsdo_uvlm.uvlm_preprocessing.utils.enum import *

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
                mesh[i, :, :, 2] = offset
            else:
                # mesh[i, :, :, 2] = mesh[i - 1, :, :, 2] + offset
                mesh[i, :, :, 2] = mesh[0, :, :, 2] 
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
        self.parameters.declare('h_stepsize')

    def define(self):
        num_times = self.parameters['num_times']

        h_stepsize = self.parameters['h_stepsize']

        ####################################
        # Create parameters
        ####################################
        for data in AcStates_vlm:
            print('{:15} = {}'.format(data.name, data.value))
            name = data.name
            string_name = data.value
            variable = self.create_input(string_name,
                                         val=AcStates_val_dict[string_name])

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
            wing_val = generate_simple_mesh(nx, ny, num_times, offset=i)
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

        # Create Model containing integrator
        ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching', num_times, display='default', visualization='None')

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict), 'subgroup', ['*'])
        # self.add_design_variable('wing_gamma_b')
        # dummy = self.declare_variable('dummy')
        # self.register_output('dummy_o',dummy+1)
        # self.add_objective('dummy_o')


if __name__ == "__main__":
    # Script to create optimization problem
    # be = 'csdl_om'
    be = 'csdl_lite'
    make_video = 1

    nx = 3
    ny = 4

    # surface_names = ['wing','wing_1']
    # surface_shapes = [(nx, ny, 3),(nx, ny-1, 3)]
    surface_names=['wing']
    surface_shapes=[(nx, ny, 3)]
    # h_stepsize = delta_t = 1/(2**0.5)
    h_stepsize = delta_t = 1.5
    # h_stepsize = delta_t = 1
    nt = 4
    
    if be == 'csdl_om':
        sim = csdl_om.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')
    if be == 'csdl_lite':
        sim = csdl_lite.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')
    sim.run()
    # sim.check_partials(compact_print=True)
    print('simulation time is', time.time() - t_start)
    print('#' * 50, 'print states', '#' * 50)
    print('wing_gamma_w_int\n', sim['wing_gamma_w_int'])
    print('wing_wake_coords_int\n', sim['wing_wake_coords_int'])

    print('#' * 50, 'print wings', '#' * 50)
    print('wing', sim['wing'])
    print('wing_gamma_w_int[-1]\n', sim['wing_gamma_w_int'][-1,:,:])

    # print('#' * 50, 'print wing_wake_total_vel', '#' * 50)
    # print('wing_wake_total_vel', sim['wing_wake_total_vel'])

    if make_video == 1:
        axs = Axes(
            xrange=(0, 35),
            yrange=(-10, 10),
            zrange=(-3, 4),
        )
        video = Video("spider.gif", duration=10, backend='ffmpeg')
        for i in range(nt - 1):
            vp = Plotter(
                bg='beige',
                bg2='lb',
                # axes=0,
                #  pos=(0, 0),
                offscreen=False,
                interactive=1)
            # Any rendering loop goes here, e.g.:
            for surface_name in surface_names:
                vps = Points(np.reshape(sim[surface_name][i, :, :, :], (-1, 3)),
                            r=8,
                            c='red')
                vp += vps
                vp += __doc__
                vps = Points(np.reshape(sim[surface_name+'_wake_coords_int'][i, 0:i, :, :],
                                        (-1, 3)),
                            r=8,
                            c='blue')
                vp += vps
                vp += __doc__
            # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
            # video.action(cameras=[cam1, cam1])
            vp.show(axs, elevation=-60, azimuth=-0,
                    axes=False)  # render the scene
            video.addFrame()  # add individual frame
            # time.sleep(0.1)
            # vp.interactive().close()
            vp.closeWindow()
        vp.closeWindow()
        video.close()  # merge all the recorded frames
    
    # sim.visualize_implementation()
    partials = sim.check_partials(compact_print=True)
    # sim.prob.check_totals(compact_print=True)