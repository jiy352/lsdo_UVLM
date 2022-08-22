from ast import Subscript
# from csdl_om import Simulator
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
        self.parameters.declare('num_nodes', default=10) 
        self.parameters.declare('delta_t', default=0.5) 
        self.parameters.declare('rot_y_rate', default=np.deg2rad(6)) 

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        # get num_nodes from surface shape
        num_nodes = self.parameters['num_nodes']
        delta_t  =self.parameters['delta_t']
        rot_y_rate  =self.parameters['rot_y_rate']

        # add_input name and shapes
        initial_mesh_names = [
            x + '_initial_mesh' for x in self.parameters['surface_names']
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

            initial_mesh = self.declare_variable(initial_mesh_name, shape=surface_shape)
            deg_temp_1 = np.linspace(-15,  20, int(num_nodes/4)).tolist()
            deg_temp_2 = np.linspace( 15, -20, int(num_nodes/4)).tolist()
            deg = (deg_temp_1 + deg_temp_2) * 2
            r_val = R.from_euler('y', deg, degrees=True)
            r  = self.create_input('R_mat',val=r_val.as_matrix())

            mesh = csdl.einsum(r, initial_mesh,subscripts='kij, lmj->klmi')
            self.register_output(surface_name, mesh)




if __name__ == "__main__":
    import pyvista as pv

    simulator_name = 'csdl_om'
    # simulator_name = 'csdl_lite'

    nt = 32
    num_pts_chord = 3
    num_pts_span = 4

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

    mesh_org = generate_simple_mesh(num_pts_chord, num_pts_span, 10)

    surface_names = ['wing']

    surface_shapes = [(num_pts_chord, num_pts_span, 3),
                      ]
    model_1 = Model()

    wing_mesh = generate_simple_mesh(num_pts_chord,
                                       num_pts_span)


    wing_1_inputs = model_1.create_input('wing_initial_mesh', val=wing_mesh)



    model_1.add(ActuationModel(surface_names=surface_names,
                                      surface_shapes=surface_shapes,
                                      num_nodes=nt,
                                      ),
                name='ActuationModel')



    if simulator_name == 'csdl_om':

        sim = Simulator(model_1)

        sim.run()
        # sim.prob.check_partials(compact_print=True)
        partials = sim.prob.check_partials(compact_print=True, out_stream=None)
        sim.assert_check_partials(partials, 1e-5, 1e-7)
        # sim.visualize_implementation()
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)

    # elif simulator_name == 'csdl_lite':
    #     sim = csdl_lite.Simulator(model_1)

    #     sim.run()
    #     sim.check_partials(compact_print=True)


    ############################################
    # Plot the lifting surfaces
    ############################################
    mesh = sim['wing']
    pv.global_theme.axes.show = True
    pv.global_theme.font.label_size = 1
    p = pv.Plotter()
    for i in range(nt):
        x = mesh[i,:, :, 0]
        y = mesh[i,:, :, 1]
        z = mesh[i,:, :, 2]

        # xw = sim['wing_wake_coords'][0, :, :, 0]
        # yw = sim['wing_wake_coords'][0, :, :, 1]
        # zw = sim['wing_wake_coords'][0, :, :, 2]


        grid = pv.StructuredGrid(x, y, z)
        # grid_1 = pv.StructuredGrid(x_1, y_1, z_1)

        
        p.add_mesh(grid, show_edges=True, opacity=.5)

    p.camera.view_angle = 60.0
    p.add_axes_at_origin(labels_off=True, line_width=5)

    p.show()
