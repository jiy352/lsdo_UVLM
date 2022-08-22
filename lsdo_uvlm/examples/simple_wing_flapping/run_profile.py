import time

import matplotlib.pyplot as plt
import openmdao.api as om
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_system_free import ODESystemModel
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_profile_outputs import ProfileSystemModel
# from lsdo_uvlm.examples.simple_wing_constant_aoa_sls.plunging_system_free import ODESystemModel

from ozone.api import ODEProblem
import csdl
# import csdl_om
# import csdl_lite
import numpy as np
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show


from lsdo_uvlm.uvlm_preprocessing.utils.enum_flapping import *
from lsdo_uvlm.uvlm_preprocessing.actuation_model_temp import ActuationModel

import cProfile, pstats, io

def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

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
        
        # self.add_profile_output('wingdummy',shape=(3, 3))
        # self.add_profile_output('wing_gamma_b',shape=(3, 3))
        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.set_ode_system(ODESystemModel)
        # self.set_profile_system(ProfileSystemModel)


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
            name = data.name
            string_name = data.value            
            print('{:15} = {},shape{}'.format(data.name, data.value,AcStates_val_dict[string_name].shape))

            variable = self.create_input(string_name,
                                         val=AcStates_val_dict[string_name])

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
            wing_val = generate_simple_mesh(nx, ny, offset=0)
            wing = self.create_input(initial_mesh_names[i], wing_val)

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
        self.add(ActuationModel(surface_names=surface_names, surface_shapes=surface_shapes, num_nodes=nt-1),'actuation_temp')

        # Create Model containing integrator
        ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching', num_times, display='default', visualization='None')

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict), 'subgroup')
        self.add(ProfileSystemModel(**profile_params_dict),'profile_outputs')
        # dummy = self.declare_variable('dummy')
        # self.register_output('dummy_o',dummy+1)
        # self.add_objective('dummy_o')


if __name__ == "__main__":
    # Script to create optimization problem
    be = 'python_csdl_backend'
    # be = 'csdl_lite'
    make_video = 1

    nx = 3
    ny = 4

    # surface_names = ['wing','wing_1']
    # surface_shapes = [(nx, ny, 3),(nx, ny-1, 3)]
    surface_names=['wing']
    surface_shapes=[(nx, ny, 3)]
    # h_stepsize = delta_t = 1/(2**0.5)
    h_stepsize = delta_t = 0.5
    # h_stepsize = delta_t = 1
    # nt = 33
    nt = 21
    
    if be == 'csdl_om':
        import csdl_om
        sim = csdl_om.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')
    if be == 'python_csdl_backend':
        import python_csdl_backend
        sim = python_csdl_backend.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')
    if be == 'csdl_lite':
        import csdl_lite
        sim = csdl_lite.Simulator(RunModel(num_times=nt - 1,h_stepsize=h_stepsize), mode='rev')
    t_start = time.time()
    def main(sim):
        sim.run()
    cProfile.run('main(sim)',"output.dat")

    with open("output_time.txt","w") as f:
        p = pstats.Stats("output.dat",stream=f)
        p.sort_stats("time").print_stats()
    
    with open("output_call.txt","w") as f:
        p = pstats.Stats("output.dat",stream=f)
        p.sort_stats("calls").print_stats()    

