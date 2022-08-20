import matplotlib.pyplot as plt
import openmdao.api as om
from ozone2.api import ODEProblem, Wrap
from lsdo_uvlm.uvlm_system.uvlm_system import UVLMSystem
import csdl_om
import numpy as np
import csdl

from lsdo_uvlm.uvlm_system.wake_rollup.combine_gamma_w_op import CombineGammaW
from lsdo_uvlm.uvlm_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp
from lsdo_uvlm.uvlm_preprocessing.adapter_comp import AdapterComp

# from lsdo_uvlm.uvlm_preprocessing.utils.enum import *
from lsdo_uvlm.uvlm_system.solve_circulations.solve_group_op import SolveMatrix
from lsdo_uvlm.uvlm_system.wake_rollup.seperate_gamma_b import SeperateGammab
from lsdo_uvlm.uvlm_system.wake_rollup.compute_wake_total_vel import ComputeWakeTotalVel

from lsdo_uvlm.uvlm_outputs.compute_force.compute_outputs_group import Outputs

class ProfileSystemModel(csdl.Model):
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
        num_nodes = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']

        # set conventional names
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        # set shapes
        bd_vortex_shapes = surface_shapes
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)
        ode_surface_shapes = [(num_nodes, ) + item for item in surface_shapes]
        # wake_vortex_pts_shapes = [tuple((item[0],nt, item[2], 3)) for item in ode_surface_shapes]
        # wake_vel_shapes = [(n,x[1] * x[2], 3) for x in wake_vortex_pts_shapes]
        ode_bd_vortex_shapes = ode_surface_shapes
        gamma_w_shapes = [tuple((num_nodes,nt-1, item[2]-1)) for item in ode_surface_shapes]


        ###################################
        # Declare the inputs
        ###################################

        '''1. add a module here to compute surface_gamma_b, given mesh and ACstates'''
        # 1.1.1 declare the ode parameter surface for the current time step
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface = self.declare_variable(surface_name, shape=(num_nodes, nx, ny, 3))
        # 1.1.2 from the declared surface mesh, compute 6 preprocessing outputs
        # surface_bd_vtx_coords,coll_pts,l_span,l_chord,s_panel,bd_vec_all
        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=ode_surface_shapes),
                 name='MeshPreprocessing_comp')
        # 1.2.1 declare the ode parameter AcStates for the current time step
        u = self.declare_variable('u',  shape=(num_nodes,1))
        v = self.declare_variable('v',  shape=(num_nodes,1))
        w = self.declare_variable('w',  shape=(num_nodes,1))
        p = self.declare_variable('p',  shape=(num_nodes,1))
        q = self.declare_variable('q',  shape=(num_nodes,1))
        r = self.declare_variable('r',  shape=(num_nodes,1))
        theta = self.declare_variable('theta',  shape=(num_nodes,1))
        psi = self.declare_variable('psi',  shape=(num_nodes,1))
        x = self.declare_variable('x',  shape=(num_nodes,1))
        y = self.declare_variable('y',  shape=(num_nodes,1))
        z = self.declare_variable('z',  shape=(num_nodes,1))
        phiw = self.declare_variable('phiw',  shape=(num_nodes,1))
        gamma = self.declare_variable('gamma',  shape=(num_nodes,1))
        psiw = self.declare_variable('psiw',  shape=(num_nodes,1))

        ###################################
        # Declare the states
        ###################################

        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            val = np.zeros((num_nodes, nt - 1, ny - 1))
            surface_name = surface_names[i]

            surface_gamma_w_name = surface_name + '_gamma_w_int'
            surface_dgammaw_dt_name = surface_name + '_dgammaw_dt'
            surface_gamma_b_name = surface_name +'_gamma_b'
            surface_wake_coords_name = surface_name + '_wake_coords_int'
            #######################################
            #states 1
            #######################################

            surface_gamma_w = self.declare_variable(surface_gamma_w_name,
                                                    shape=val.shape)
            surface_wake_coords = self.declare_variable(surface_wake_coords_name, shape=(num_nodes, nt - 1, ny, 3))
            # self.print_var(theta)
            surface_dummy = csdl.expand(csdl.reshape(theta,(num_nodes,)),(val.shape),'i->ijk')+surface_gamma_w
            self.register_output(surface_name+'dummy', surface_dummy)

        ###################################
        # solve for gamma_b
        ###################################
        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
        )
        self.add(m, name='adapter_comp')

        self.add(CombineGammaW(surface_names=surface_names, surface_shapes=ode_surface_shapes, n_wake_pts_chord=nt-1),
            name='combine_gamma_w')

        self.add(SolveMatrix(n_wake_pts_chord=nt-1,
                                surface_names=surface_names,
                                bd_vortex_shapes=ode_surface_shapes,
                                delta_t=delta_t),
                    name='solve_gamma_b_group')

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes),
                 name='seperate_gamma_b')

        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]

        # compute lift and drag
        sub = Outputs(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option='auto',
            eval_pts_location=0.25,
            sprs=None,
            coeffs_aoa=None,
            coeffs_cd=None,
            n_wake_pts_chord=nt-1,
        )
        self.add(sub, name='VLM_outputs')


if __name__ == "__main__":
    # from lsdo_uvlm.uvlm_preprocessing.utils.enum import *
    import enum
    from csdl_om import Simulator
    import csdl_lite
    # simulator_name = 'csdl_lite'
    simulator_name = 'csdl_om'
    num_nodes = 1

    class AcStates_vlm(enum.Enum):
        u = 'u'
        v = 'v'
        w = 'w'
        p = 'p'
        q = 'q'
        r = 'r'
        theta = 'theta'
        psi = 'psi'
        x = 'x'
        y = 'y'
        z = 'z'
        phiw = 'phiw'
        gamma = 'gamma'
        psiw = 'psiw'
        # rho = 'rho'


    # v_inf = np.array([2, 2, 2, 2, ])
    alpha_deg =10
    alpha = alpha_deg / 180 * np.pi
    # vx = -v_inf * np.cos(alpha)
    # vz = -v_inf * np.sin(alpha)

    AcStates_val_dict = {
        AcStates_vlm.u.value: np.ones((num_nodes, 1))* np.cos(alpha),
        AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.w.value: np.ones((num_nodes, 1))* np.sin(alpha),
        AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.theta.value: np.ones((num_nodes, 1))*alpha,
        AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.z.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.phiw.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.gamma.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.psiw.value: np.zeros((num_nodes, 1)),
        # AcStates_vlm.rho.value: np.ones((num_nodes, 1)) * 0.96,
    }

    def generate_simple_mesh(nx, ny, n_wake_pts_chord=None,offset=0):
        if n_wake_pts_chord == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((n_wake_pts_chord, nx, ny, 3))
            for i in range(n_wake_pts_chord):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0. + offset
        return mesh

    nx=3
    ny=4
    surface_names=['wing','wing_1']

    surface_shapes = [(nx,ny,3),(nx,ny,3)]

    wing_val = generate_simple_mesh(nx, ny, 1)
    wing_val_1 = generate_simple_mesh(nx,ny, 1, 1)



    model_1 = csdl.Model()
    wing = model_1.create_input(surface_names[0], wing_val)
    wing = model_1.create_input(surface_names[1], wing_val_1)
    for data in AcStates_vlm:
        print('{:15} = {}'.format(data.name, data.value))
        name = data.name
        string_name = data.value
        variable = model_1.create_input(string_name,
                                        val=AcStates_val_dict[string_name])
    
    m = ODESystemModel(surface_names=surface_names,surface_shapes=surface_shapes,delta_t=1,nt=5)
    model_1.add(m,'ode')
    if simulator_name == 'csdl_om':

        sim = Simulator(model_1)

        sim.run()
        # sim.prob.check_partials(compact_print=True)
        # partials = sim.prob.check_partials(compact_print=True, out_stream=None)
        # sim.assert_check_partials(partials, 1e-5, 1e-7)
        sim.visualize_implementation()
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)

    elif simulator_name == 'csdl_lite':
        sim = csdl_lite.Simulator(model_1)

        sim.run()
        sim.check_partials(compact_print=True)
