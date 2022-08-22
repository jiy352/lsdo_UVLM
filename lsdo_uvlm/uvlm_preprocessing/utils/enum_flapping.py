import enum
import numpy as np

# num_nodes = 32
# num_nodes = 4
# num_nodes = 20
num_nodes = 12
# num_nodes=3
# num_nodes=3



# v_inf = np.array([2, 2, 2, 2, ])
alpha_deg =0
alpha = alpha_deg / 180 * np.pi
deg = -np.deg2rad(10)

dir = ([1, ] * 8 + [-1, ] * 8 + [1, ] * 8 + [-1, ] * 8 )[:num_nodes]
# dir = [1, ] * num_nodes
# dir = [1, ] * 8 + [-1, ] * 4
# dir = [1, ] * 8 + [-1, ] * 8
# dir = [1, ] * 8 + [-1, ] * 8+ [1, ] * 4

AcStates_val_dict = {
    'u': np.ones((num_nodes, 1))* np.cos(alpha)*2,
    'v': np.zeros((num_nodes, 1)),
    'w': np.ones((num_nodes, 1))* np.sin(alpha),
    'p': np.zeros((num_nodes, 1)),
    'q': np.array(dir)*deg,
    'r': np.zeros((num_nodes, 1)),
    'theta': np.ones((num_nodes, 1))*alpha,
    'psi': np.zeros((num_nodes, 1)),
    'x': np.zeros((num_nodes, 1)),
    'y': np.zeros((num_nodes, 1)),
    'z': np.zeros((num_nodes, 1)),
    'phiw': np.zeros((num_nodes, 1)),
    'gamma': np.zeros((num_nodes, 1)),
    'psiw': np.zeros((num_nodes, 1)),
}