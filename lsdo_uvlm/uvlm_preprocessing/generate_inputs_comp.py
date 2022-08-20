from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from fluids import atmosphere as atmosphere
from lsdo_UVLM.uvlm_preprocessing.utils.enum import *


class GenerateInputsComp(Model):
    """
    An adapter component that creates 15 inputs related to the lifting surface states

    parameters
    ----------
    u[num_nodes,1] : csdl array
        vx of the body
    v[num_nodes,1] : csdl array
        vy of the body
    w[num_nodes,1] : csdl array
        vz of the body

    p[num_nodes,1] : csdl array
        omega_x of the body
    q[num_nodes,1] : csdl array
        omega_y of the body
    r[num_nodes,1] : csdl array
        omega_z of the body

    phi[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: p=\dot{phi}
    theta[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: q=\dot{theta}
    psi[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: r=\dot{psi}

    x[num_nodes,1] : csdl array
        omega_x of the body
    y[num_nodes,1] : csdl array
        omega_y of the body
    z[num_nodes,1] : csdl array
        omega_z of the body

    phiw[num_nodes,1] : csdl array
        omega_x of the body
    gamma[num_nodes,1] : csdl array
        omega_y of the body
    psiw[num_nodes,1] : csdl array
        omega_z of the body    

    collocation points

    Returns
    -------

    """
    def initialize(self):
        pass

    def define(self):
        # add_input
        for data in AcStates_vlm:
            print('{:15} = {}'.format(data.name, data.value))
            name = data.name
            string_name = data.value
            variable = self.create_input(string_name,
                                         val=AcStates_val_dict[string_name])


if __name__ == "__main__":

    import csdl_lite
    simulator_name = 'csdl_om'
    # simulator_name = 'csdl_lite'

    n_wake_pts_chord = 2
    num_pts_chord = 3
    num_pts_span = 4
    from lsdo_UVLM.uvlm_preprocessing.utils.enum import *
    from lsdo_UVLM.uvlm_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp
    from lsdo_UVLM.uvlm_preprocessing.adapter_comp import AdapterComp

    # add the upstream mesh preprocessing comp

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
    surface_shapes = [(n_wake_pts_chord, num_pts_chord, num_pts_span, 3),
                      (n_wake_pts_chord, num_pts_chord + 1, num_pts_span + 1,
                       3)]
    model_1 = Model()

    wing_1_mesh = generate_simple_mesh(n_wake_pts_chord, num_pts_chord,
                                       num_pts_span)
    wing_2_mesh = generate_simple_mesh(n_wake_pts_chord, num_pts_chord + 1,
                                       num_pts_span + 1)

    # wing_1_inputs = model_1.create_input('wing_1', val=wing_1_mesh)
    # wing_2_inputs = model_1.create_input('wing_2', val=wing_2_mesh)

    model_1 = Model()

    # add the current comp
    model_1.add(GenerateInputsComp(), name='GenerateInputsComp')
    model_1.add(AdapterComp(surface_names=surface_names,
                            surface_shapes=surface_shapes),
                name='AdapterComp')

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
