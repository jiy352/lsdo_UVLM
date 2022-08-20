

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,

# :::::::y-->gamma_b,aic_bd_proj,M_mat,gamma_w,b,:::::::

# _00Wn linear_combination_eval_py_pb
path_to__00Wm = py_p_00Wm.copy()
path_to_b = py_pb.copy()

# _00Wl linear_combination_eval_p_00Wm_p_00Wk
path_to__00Wg = path_to__00Wm@p_00Wm_p_00Wg
path_to__00Wk = path_to__00Wm@p_00Wm_p_00Wk

# _00Wf einsum_eval_p_00Wg_pgamma_b
_00Wf_temp_einsum = _00Wf_partial_func(aic_bd_proj, gamma_b)
p_00Wg_paic_bd_proj = _00Wf_temp_einsum[0]
p_00Wg_pgamma_b = _00Wf_temp_einsum[1]
path_to_aic_bd_proj = path_to__00Wg@p_00Wg_paic_bd_proj
path_to_gamma_b = path_to__00Wg@p_00Wg_pgamma_b

# _00Wj einsum_eval_p_00Wk_pM_mat
_00Wj_temp_einsum = _00Wj_partial_func(M_mat, _00Wi)
p_00Wk_p_00Wi = _00Wj_temp_einsum[0]
p_00Wk_pM_mat = _00Wj_temp_einsum[1]
path_to__00Wi = path_to__00Wk@p_00Wk_p_00Wi
path_to_M_mat = path_to__00Wk@p_00Wk_pM_mat

# _00Wh reshape_eval_p_00Wi_pgamma_w
path_to_gamma_w = path_to__00Wi@p_00Wi_pgamma_w
dy_dgamma_b = path_to_gamma_b.copy()
dy_daic_bd_proj = path_to_aic_bd_proj.copy()
dy_dM_mat = path_to_M_mat.copy()
dy_dgamma_w = path_to_gamma_w.copy()
dy_db = path_to_b.copy()