

# GC:y,-->gamma_b,MTX,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,MTX,b,

# :::::::y-->gamma_b,MTX,b,:::::::

# _00eC linear_combination_eval_py_pb
path_to__00eB = py_p_00eB.copy()
path_to_b = py_pb.copy()

# _00eA einsum_eval_p_00eB_pgamma_b
_00eA_temp_einsum = _00eA_partial_func(MTX, gamma_b)
p_00eB_pMTX = _00eA_temp_einsum[0]
p_00eB_pgamma_b = _00eA_temp_einsum[1]
path_to_MTX = path_to__00eB@p_00eB_pMTX
path_to_gamma_b = path_to__00eB@p_00eB_pgamma_b
dy_dgamma_b = path_to_gamma_b.copy()
dy_dMTX = path_to_MTX.copy()
dy_db = path_to_b.copy()