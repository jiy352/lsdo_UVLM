

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,

# :::::::y-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,:::::::

# _00GH linear_combination_eval_py_pb
path_to__00GG = py_p_00GG.copy()
path_to_b = py_pb.copy()

# _00GF linear_combination_eval_p_00GG_p_00GE
path_to__00GA = path_to__00GG@p_00GG_p_00GA
path_to__00GE = path_to__00GG@p_00GG_p_00GE

# _00Gz einsum_eval_p_00GA_pgamma_b
_00Gz_temp_einsum = _00Gz_partial_func(aic_bd_proj, gamma_b)
p_00GA_paic_bd_proj = _00Gz_temp_einsum[0]
p_00GA_pgamma_b = _00Gz_temp_einsum[1]
path_to_aic_bd_proj = path_to__00GA@p_00GA_paic_bd_proj
path_to_gamma_b = path_to__00GA@p_00GA_pgamma_b

# _00GD einsum_eval_p_00GE_pM_mat
_00GD_temp_einsum = _00GD_partial_func(M_mat, _00GC)
p_00GE_p_00GC = _00GD_temp_einsum[0]
p_00GE_pM_mat = _00GD_temp_einsum[1]
path_to__00GC = path_to__00GE@p_00GE_p_00GC
path_to_M_mat = path_to__00GE@p_00GE_pM_mat

# _00GB reshape_eval_p_00GC_pgamma_w
path_to_gamma_w = path_to__00GC@p_00GC_pgamma_w
dy_dgamma_b = path_to_gamma_b.copy()
dy_daic_bd_proj = path_to_aic_bd_proj.copy()
dy_dM_mat = path_to_M_mat.copy()
dy_dgamma_w = path_to_gamma_w.copy()
dy_db = path_to_b.copy()