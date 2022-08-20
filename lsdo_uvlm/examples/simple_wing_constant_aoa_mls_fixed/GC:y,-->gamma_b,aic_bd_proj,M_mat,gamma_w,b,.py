

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,

# :::::::y-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,:::::::

# _00VH linear_combination_eval_py_pb
path_to__00VG = py_p_00VG.copy()
path_to_b = py_pb.copy()

# _00VF linear_combination_eval_p_00VG_p_00VE
path_to__00VA = path_to__00VG@p_00VG_p_00VA
path_to__00VE = path_to__00VG@p_00VG_p_00VE

# _00Vz einsum_eval_p_00VA_pgamma_b
_00Vz_temp_einsum = _00Vz_partial_func(aic_bd_proj, gamma_b)
p_00VA_paic_bd_proj = _00Vz_temp_einsum[0]
p_00VA_pgamma_b = _00Vz_temp_einsum[1]
path_to_aic_bd_proj = path_to__00VA@p_00VA_paic_bd_proj
path_to_gamma_b = path_to__00VA@p_00VA_pgamma_b

# _00VD einsum_eval_p_00VE_pM_mat
_00VD_temp_einsum = _00VD_partial_func(M_mat, _00VC)
p_00VE_p_00VC = _00VD_temp_einsum[0]
p_00VE_pM_mat = _00VD_temp_einsum[1]
path_to__00VC = path_to__00VE@p_00VE_p_00VC
path_to_M_mat = path_to__00VE@p_00VE_pM_mat

# _00VB reshape_eval_p_00VC_pgamma_w
path_to_gamma_w = path_to__00VC@p_00VC_pgamma_w
dy_dgamma_b = path_to_gamma_b.copy()
dy_daic_bd_proj = path_to_aic_bd_proj.copy()
dy_dM_mat = path_to_M_mat.copy()
dy_dgamma_w = path_to_gamma_w.copy()
dy_db = path_to_b.copy()