# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy import random
from numpy.core.fromnumeric import shape, size
from numpy.random import gamma


class CombineGammaW(Model):
    """
    seperate the whole solution vector gamma_b
    corresponding to different lifting surfaces

    parameters
    ----------
    gamma_b

    Returns
    -------
    surface_name+'_gamma_b' : csdl array
    """
    def initialize(self):

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('n_wake_pts_chord')


    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        num_nodes = surface_shapes[0][0]

        num_nodes = surface_shapes[0][0]
        surface_gamma_w_shapes =  [tuple((item[0], n_wake_pts_chord, item[2]-1)) for item in surface_shapes]
        # print('surface_shapes',surface_shapes)
        # print('(num_nodes, n_wake_pts_chord,)',(num_nodes, n_wake_pts_chord,))
        # print('tuple(sum((i[2] - 1) for i in surface_shapes)',((i[2] - 1) for i in surface_shapes))
        gamma_w_shape = (num_nodes, n_wake_pts_chord,)+ (sum((i[2] - 1) for i in surface_shapes),)

        # sum of system_shape with all the nx's and ny's
        gamma_w = self.create_output('gamma_w', shape=gamma_w_shape)

        start = 0
        for i in range(len(surface_shapes)):
            surface_shape = surface_shapes[i]

            ny = surface_shape[2]
            delta = ny-1
            
            surface_gamma_w = self.declare_variable(surface_names[i]+'_gamma_w_int',shape=surface_gamma_w_shapes[i])

            gamma_w[:, :, start:start+delta] = surface_gamma_w
            start += delta


if __name__ == "__main__":

    pass