

# GC:y,-->gamma_b,MTX,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,MTX,b,

# :::::::y-->gamma_b,MTX,b,:::::::

# _00dW linear_combination_eval_py_pb
path_to__00dV = py_p_00dV.copy()
path_to_b = py_pb.copy()

# _00dU einsum_eval_p_00dV_pgamma_b
_00dU_temp_einsum = _00dU_partial_func(MTX, gamma_b)
p_00dV_pMTX = _00dU_temp_einsum[0]
p_00dV_pgamma_b = _00dU_temp_einsum[1]
path_to_MTX = path_to__00dV@p_00dV_pMTX
path_to_gamma_b = path_to__00dV@p_00dV_pgamma_b
dy_dgamma_b = path_to_gamma_b.copy()
dy_dMTX = path_to_MTX.copy()
dy_db = path_to_b.copy()