

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,

# :::::::y-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,:::::::

# _00hd linear_combination_eval_py_pb
path_to__00hc = py_p_00hc.copy()
path_to_b = py_pb.copy()

# _00hb linear_combination_eval_p_00hc_p_00ha
path_to__00h6 = path_to__00hc@p_00hc_p_00h6
path_to__00ha = path_to__00hc@p_00hc_p_00ha

# _00h5 einsum_eval_p_00h6_pgamma_b
_00h5_temp_einsum = _00h5_partial_func(aic_bd_proj, gamma_b)
p_00h6_paic_bd_proj = _00h5_temp_einsum[0]
p_00h6_pgamma_b = _00h5_temp_einsum[1]
path_to_aic_bd_proj = path_to__00h6@p_00h6_paic_bd_proj
path_to_gamma_b = path_to__00h6@p_00h6_pgamma_b

# _00h9 einsum_eval_p_00ha_pM_mat
_00h9_temp_einsum = _00h9_partial_func(M_mat, _00h8)
p_00ha_p_00h8 = _00h9_temp_einsum[0]
p_00ha_pM_mat = _00h9_temp_einsum[1]
path_to__00h8 = path_to__00ha@p_00ha_p_00h8
path_to_M_mat = path_to__00ha@p_00ha_pM_mat

# _00h7 reshape_eval_p_00h8_pgamma_w
path_to_gamma_w = path_to__00h8@p_00h8_pgamma_w
dy_dgamma_b = path_to_gamma_b.copy()
dy_daic_bd_proj = path_to_aic_bd_proj.copy()
dy_dM_mat = path_to_M_mat.copy()
dy_dgamma_w = path_to_gamma_w.copy()
dy_db = path_to_b.copy()