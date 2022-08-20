import numpy as np
from scipy.spatial.transform import Rotation as R

import pyvista as pv
nx = 3
ny = 4
nt = 11

# deg = (np.arange(10)*10)
deg = np.linspace(-20,20,nt).tolist()
r = R.from_euler('y', deg, degrees=True)



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

mesh_org = generate_simple_mesh(nx, ny, nt)

mesh = np.einsum('kij, klmj->klmi',r.as_matrix(), mesh_org)



############################################
# Plot the lifting surfaces
############################################
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