import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from math import pi


def generate_eel_carling_vlm(num_pts_L,num_pts_R,L,s_1_ind,s_2_ind):
    w1 = 0.04*L
    w2 = 0.01*L
    a = 0.55*L
    b = 0.08*L

    s1 = 0.04 * L
    s2 = 0.95 * L

    width = np.zeros(num_pts_L)
    height = np.zeros(num_pts_L)
    x_1 = (1-np.cos(np.linspace(0, np.pi/2,s_1_ind)))/1*s1
    x_2 = np.linspace(s1, s2, int(s_2_ind-s_1_ind))
    x_3 = np.linspace(s2, L, int(num_pts_L-s_2_ind))

    x = np.concatenate((x_1,x_2,x_3))
    width[:s_1_ind] = (2*w1*x[:s_1_ind]-(x[:s_1_ind]**2))**0.5
    width[s_1_ind:s_2_ind] = w2 - (w2-w1)*((x[s_1_ind:s_2_ind]-s2)/(s2-s1))**2
    width[s_2_ind:] = w2*(L-x[s_2_ind:])/(L-s2)

    height = b*(1-((x-a)/a)**2)**0.5
    plt.plot(x, width)
    plt.plot(x, height)
    plt.legend(['width', 'height'])
    plt.xlabel('x [m]')
    plt.ylabel('width/height [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('eel_verification_mesh.pdf')
    plt.show()

    print("width", width.shape, width)
    print("height", height.shape, height)

    u=0.     #x-position of the center
    v=0.   #y-position of the center
    theta = np.linspace(0, 2*pi, num_pts_R)
    x_all = np.outer(x, np.ones(num_pts_R))

    zdata = np.outer(np.ones(num_pts_L)*height,np.sin(theta))
    xdata = x_all
    ydata = np.outer(np.ones(num_pts_L)*width,np.cos(theta))
    filename = 'eel_verification_mesh_ugly.vtk'
    shape = (num_pts_L,num_pts_R)
    x = x_all.reshape(shape).T
    y = ydata.reshape(shape).T
    z = zdata.reshape(shape).T
    

    # grid = pv.StructuredGrid(x_final, y_final, z_final)
    grid = pv.StructuredGrid(x, y, z)
    # grid.plot(show_edges=True, show_bounds=True, show_grid=True, show_axes=True)
    # grid.save(filename)

    # grid = grid.compute_cell_sizes(length=False, volume=False)
    # area_vec = grid.cell_data['Area']
    # area = np.sum(area_vec)

    # print(area)
    # exit()

    return grid,height


def get_connectivity_matrix_from_pyvista(grid):
    cell_idx = np.arange(grid.n_cells)
    connectivitiy_mtx = np.zeros((grid.n_cells, 4))
    points = np.array(grid.points)
    for i in cell_idx:
        # print(i)
        cell_current = cell_idx[i]
        cell = grid.GetCell(cell_current)
        point_indices = cell.GetPointIds()
        pids = pv.vtk_id_list_to_array(cell.GetPointIds())

        connectivitiy_mtx[i,0] = pids[3]
        connectivitiy_mtx[i,1] = pids[0]
        connectivitiy_mtx[i,2] = pids[1]
        connectivitiy_mtx[i,3] = pids[2] 

        # connectivitiy_mtx[i,0] = pids[0]
        # connectivitiy_mtx[i,1] = pids[1]
        # connectivitiy_mtx[i,2] = pids[2]
        # connectivitiy_mtx[i,3] = pids[3] 
        
    element_table = points[connectivitiy_mtx.astype(int),:]
    return points, element_table, connectivitiy_mtx.astype(int)

def neighbor_cell_idx(grid):
    # grid = pv.read(filename)
    cell_idx = np.arange(grid.n_cells)
    # neighbors_idx = (np.ones((grid.n_cells, 4))*9876).astype(dtype=int)
    neighbors_idx = np.outer(cell_idx,np.ones(4))
    for i in cell_idx:
        # print(i)
        cell_current = cell_idx[i]
        cell = grid.GetCell(cell_current)
        pids = pv.vtk_id_list_to_array(cell.GetPointIds())
        neighbors = list(set(grid.extract_points(pids,)["vtkOriginalCellIds"]))

        corner_1 = set(grid.extract_points(pids[0])["vtkOriginalCellIds"])
        corner_2 = set(grid.extract_points(pids[1])["vtkOriginalCellIds"])
        corner_3 = set(grid.extract_points(pids[2])["vtkOriginalCellIds"])
        corner_4 = set(grid.extract_points(pids[3])["vtkOriginalCellIds"])
        mylist = list(corner_1)+list(corner_2)+list(corner_3)+list(corner_4)

        neighbors = set([x for x in mylist if not x in neighbors or neighbors.remove(x)])

        neighbors.discard(cell_current)

        neighbors_ = np.array(list(neighbors))
        neighbors_idx[i, :neighbors_.size] = neighbors_.astype(dtype=int)
    return neighbors_idx

def get_swimming_eel_geometry(points_undef_reshaped, y):
    num_nodes = y.shape[0]
    num_pts_L = points_undef_reshaped.shape[1]
    num_pts_R = points_undef_reshaped.shape[2]
    
    y_exp = np.einsum('ij, k -> ijk', y, np.ones(points_undef_reshaped.shape[2]))

    points_temp = np.outer(np.ones(y.shape[0]),points_undef_reshaped).reshape(num_nodes,num_pts_L,num_pts_R,3)
    points_def_reshaped = points_temp.copy() 
    points_def_reshaped[:,:,:,1] = points_temp[:,:,:,1].copy()+ y_exp
    points = points_def_reshaped.reshape((num_nodes,-1,3))

    return points



if __name__ == '__main__':
    num_pts_L = 20
    num_pts_R=15
    L = 1.
    s_1_ind = 5
    s_2_ind = 16
    pts = generate_eel_carling(num_pts_L,num_pts_R,L,s_1_ind,s_2_ind)