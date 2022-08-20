from ast import Subscript
from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
from scipy.spatial.transform import Rotation as R


class ActuationModel(Model):
    """
    Compute the actuated mesh and the mesh velocity on mesh cornor points
    for now just set mesh cornor points velocity as a input to the ode
    and use kinematic vel comp to interpolate it to the collocation points
    The inputs of this actuation model is the actuation parameters 
    (rate of rotation (size= (num_nodes,1 )), linear velocity relative to the inertia frame), and the initial_mesh (nx, ny, 3)
    The outputs are 
    mesh_velocity (num_nodes, nx, ny, 3)
    mesh (num_nodes, nx, ny, 3)

    Here, I just start with a case of a flapping tail (lin_vel = 0; rot_vel = (rx, ry, rz), where rx=rz=0 )

    parameters
    ----------
    (actuation): rate of rot
                 linear_vel (num_nodes, nx, ny, 1)  
    initial_mesh
    Returns
    -------
    mesh_velocity
    mesh
    """
    def initialize(self):

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list) #(nx, ny, 3)
        self.parameters.declare('num_nodes', types=int) 
        self.parameters.declare('delta_t', default=0.5) 

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        # get num_nodes from surface shape
        num_nodes = self.parameters['num_nodes']
        delta_t  =self.parameters['delta_t']

        # add_input name and shapes
        initial_mesh_names = [
            x + '_intial_mesh' for x in self.parameters['surface_names']
        ]

        surface_rot_vel_names = [
            x + '_act_rot_vel' for x in self.parameters['surface_names']
        ]
        surface_rot_vel_shape = (num_nodes, 3)

        surface_linear_vel_names = [
            x + '_act_lin_vel' for x in self.parameters['surface_names']
        ]

        surface_linear_vel_shapes = [
            (num_nodes, ) + x  for x in self.parameters['surface_shapes']
        ]


        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            num_pts_chord = surface_shapes[i][0]
            num_pts_span = surface_shapes[i][1 ]

            initial_mesh_name = initial_mesh_names[i]

            initial_mesh = self.declare_variable('initial_mesh_name', shape=surface_shape)

            surface_rot_vel = self.declare_variable(surface_linear_vel_names[i], shape=surface_rot_vel_shape)

            rotational_angles = surface_rot_vel*delta_t
            surface_linear_vel = self.declare_variable(surface_linear_vel_names, shape=surface_linear_vel_shapes)

            actuated_mesh = 



if __name__ == "__main__":

    import csdl_lite
    simulator_name = 'csdl_om'
    # simulator_name = 'csdl_lite'

    n_wake_pts_chord = 2
    num_pts_chord = 3
    num_pts_span = 4

    from lsdo_uvlm.uvlm_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp
    from lsdo_uvlm.uvlm_preprocessing.adapter_comp import AdapterComp
    from lsdo_uvlm.uvlm_preprocessing.utils.enum import *

    # add the upstream mesh preprocessing comp
    def generate_simple_mesh(n_wake_pts_chord, num_pts_chord, num_pts_span):
        mesh = np.zeros((n_wake_pts_chord, num_pts_chord, num_pts_span, 3))
        for i in range(n_wake_pts_chord):
            mesh[i, :, :, 0] = np.outer(np.arange(num_pts_chord),
                                        np.ones(num_pts_span))
            mesh[i, :, :, 1] = np.outer(np.arange(num_pts_span),
                                        np.ones(num_pts_chord)).T
            mesh[i, :, :, 2] = 0.
        return mesh

    surface_names = ['wing_1', 'wing_2']

    kinematic_vel_names = [x + '_kinematic_vel' for x in surface_names]
    surface_shapes = [(n_wake_pts_chord, num_pts_chord, num_pts_span, 3),
                      (n_wake_pts_chord, num_pts_chord + 1, num_pts_span + 1,
                       3)]
    model_1 = Model()

    wing_1_mesh = generate_simple_mesh(n_wake_pts_chord, num_pts_chord,
                                       num_pts_span)
    wing_2_mesh = generate_simple_mesh(n_wake_pts_chord, num_pts_chord + 1,
                                       num_pts_span + 1)

    wing_1_inputs = model_1.create_input('wing_1', val=wing_1_mesh)
    wing_2_inputs = model_1.create_input('wing_2', val=wing_2_mesh)

    create_opt = 'create_inputs'

    #creating inputs that share the same names within CADDEE
    for data in AcStates_vlm:
        print('{:15} = {}'.format(data.name, data.value))
        name = data.name
        string_name = data.value
        if create_opt == 'create_inputs':
            variable = model_1.create_input(string_name,
                                            val=AcStates_val_dict[string_name])
            # del variable
        else:
            variable = model_1.declare_variable(
                string_name, val=AcStates_val_dict[string_name])

    model_1.add(MeshPreprocessingComp(surface_names=surface_names,
                                      surface_shapes=surface_shapes),
                name='MeshPreprocessingComp')
    # add the current comp
    model_1.add(AdapterComp(surface_names=surface_names,
                            surface_shapes=surface_shapes),
                name='AdapterComp')
    model_1.add(KinematicVelocityComp(surface_names=surface_names,
                                      surface_shapes=surface_shapes),
                name='KinematicVelocityComp')
    if simulator_name == 'csdl_om':

        sim = Simulator(model_1)

        sim.run()
        # sim.prob.check_partials(compact_print=True)
        partials = sim.prob.check_partials(compact_print=True, out_stream=None)
        sim.assert_check_partials(partials, 1e-5, 1e-7)
        sim.visualize_implementation()
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)

    elif simulator_name == 'csdl_lite':
        sim = csdl_lite.Simulator(model_1)

        sim.run()
        sim.check_partials(compact_print=True)

    print('u,v,w,p,q,r\n', sim['u'], sim['v'], sim['w'], sim['p'], sim['q'],
          sim['r'])
    print('frame_vel,alpha,v_inf_sq,beta,kinematic_vel(wing 1, wing2),rho\n',
          sim['frame_vel'], sim['alpha'], sim['v_inf_sq'], sim['beta'],
          sim[kinematic_vel_names[0]], sim[kinematic_vel_names[1]], sim['rho'])
