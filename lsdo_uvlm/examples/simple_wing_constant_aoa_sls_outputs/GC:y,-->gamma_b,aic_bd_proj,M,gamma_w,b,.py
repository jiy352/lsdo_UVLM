

# GC:y,-->gamma_b,aic_bd_proj,M,gamma_w,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,aic_bd_proj,M,gamma_w,b,

# :::::::y-->gamma_b,aic_bd_proj,M,gamma_w,b,:::::::

# _00ed linear_combination_eval_py_pb
path_to__00ec = py_p_00ec.copy()
path_to_b = py_pb.copy()

# _00eb linear_combination_eval_p_00ec_p_00ea
path_to__00e8 = path_to__00ec@p_00ec_p_00e8
path_to__00ea = path_to__00ec@p_00ec_p_00ea

# _00e7 einsum_eval_p_00e8_pgamma_b
_00e7_temp_einsum = _00e7_partial_func(aic_bd_proj, gamma_b)
p_00e8_paic_bd_proj = _00e7_temp_einsum[0]
p_00e8_pgamma_b = _00e7_temp_einsum[1]
path_to_aic_bd_proj = path_to__00e8@p_00e8_paic_bd_proj
path_to_gamma_b = path_to__00e8@p_00e8_pgamma_b

# _00e9 einsum_eval_p_00ea_pgamma_w
_00e9_temp_einsum = _00e9_partial_func(M, gamma_w)
p_00ea_pM = _00e9_temp_einsum[0]
p_00ea_pgamma_w = _00e9_temp_einsum[1]
path_to_M = path_to__00ea@p_00ea_pM
path_to_gamma_w = path_to__00ea@p_00ea_pgamma_w
dy_dgamma_b = path_to_gamma_b.copy()
dy_daic_bd_proj = path_to_aic_bd_proj.copy()
dy_dM = path_to_M.copy()
dy_dgamma_w = path_to_gamma_w.copy()
dy_db = path_to_b.copy()