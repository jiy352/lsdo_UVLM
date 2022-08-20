

# GC:y,-->gamma_b,aic_bd_proj,M_mat,wing_gamma_w,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,aic_bd_proj,M_mat,wing_gamma_w,b,

# :::::::y-->gamma_b,aic_bd_proj,M_mat,wing_gamma_w,b,:::::::

# _00ht linear_combination_eval_py_pb
path_to__00hs = py_p_00hs.copy()
path_to_b = py_pb.copy()

# _00hr linear_combination_eval_p_00hs_p_00hq
path_to__00hm = path_to__00hs@p_00hs_p_00hm
path_to__00hq = path_to__00hs@p_00hs_p_00hq

# _00hl einsum_eval_p_00hm_pgamma_b
_00hl_temp_einsum = _00hl_partial_func(aic_bd_proj, gamma_b)
p_00hm_paic_bd_proj = _00hl_temp_einsum[0]
p_00hm_pgamma_b = _00hl_temp_einsum[1]
path_to_aic_bd_proj = path_to__00hm@p_00hm_paic_bd_proj
path_to_gamma_b = path_to__00hm@p_00hm_pgamma_b

# _00hp einsum_eval_p_00hq_pM_mat
_00hp_temp_einsum = _00hp_partial_func(M_mat, _00ho)
p_00hq_p_00ho = _00hp_temp_einsum[0]
p_00hq_pM_mat = _00hp_temp_einsum[1]
path_to__00ho = path_to__00hq@p_00hq_p_00ho
path_to_M_mat = path_to__00hq@p_00hq_pM_mat

# _00hn reshape_eval_p_00ho_pwing_gamma_w
path_to_wing_gamma_w = path_to__00ho@p_00ho_pwing_gamma_w
dy_dgamma_b = path_to_gamma_b.copy()
dy_daic_bd_proj = path_to_aic_bd_proj.copy()
dy_dM_mat = path_to_M_mat.copy()
dy_dwing_gamma_w = path_to_wing_gamma_w.copy()
dy_db = path_to_b.copy()