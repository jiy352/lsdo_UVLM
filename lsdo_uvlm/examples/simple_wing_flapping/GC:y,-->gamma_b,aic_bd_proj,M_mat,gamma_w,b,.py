

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,

# :::::::y-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,:::::::

# _00hE linear_combination_eval_py_pb
path_to__00hD = py_p_00hD.copy()
path_to_b = py_pb.copy()

# _00hC linear_combination_eval_p_00hD_p_00hB
path_to__00hx = path_to__00hD@p_00hD_p_00hx
path_to__00hB = path_to__00hD@p_00hD_p_00hB

# _00hw einsum_eval_p_00hx_pgamma_b
_00hw_temp_einsum = _00hw_partial_func(aic_bd_proj, gamma_b)
p_00hx_paic_bd_proj = _00hw_temp_einsum[0]
p_00hx_pgamma_b = _00hw_temp_einsum[1]
path_to_aic_bd_proj = path_to__00hx@p_00hx_paic_bd_proj
path_to_gamma_b = path_to__00hx@p_00hx_pgamma_b

# _00hA einsum_eval_p_00hB_pM_mat
_00hA_temp_einsum = _00hA_partial_func(M_mat, _00hz)
p_00hB_p_00hz = _00hA_temp_einsum[0]
p_00hB_pM_mat = _00hA_temp_einsum[1]
path_to__00hz = path_to__00hB@p_00hB_p_00hz
path_to_M_mat = path_to__00hB@p_00hB_pM_mat

# _00hy reshape_eval_p_00hz_pgamma_w
path_to_gamma_w = path_to__00hz@p_00hz_pgamma_w
dy_dgamma_b = path_to_gamma_b.copy()
dy_daic_bd_proj = path_to_aic_bd_proj.copy()
dy_dM_mat = path_to_M_mat.copy()
dy_dgamma_w = path_to_gamma_w.copy()
dy_db = path_to_b.copy()