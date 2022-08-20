

# GC:y,-->gamma_b,aic_bd_proj,M_mat,wing_gamma_w,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,aic_bd_proj,M_mat,wing_gamma_w,b,

# :::::::y-->gamma_b,aic_bd_proj,M_mat,wing_gamma_w,b,:::::::

# _00Wc linear_combination_eval_py_pb
path_to__00Wb = py_p_00Wb.copy()
path_to_b = py_pb.copy()

# _00Wa linear_combination_eval_p_00Wb_p_00W9
path_to__00W5 = path_to__00Wb@p_00Wb_p_00W5
path_to__00W9 = path_to__00Wb@p_00Wb_p_00W9

# _00W4 einsum_eval_p_00W5_pgamma_b
_00W4_temp_einsum = _00W4_partial_func(aic_bd_proj, gamma_b)
p_00W5_paic_bd_proj = _00W4_temp_einsum[0]
p_00W5_pgamma_b = _00W4_temp_einsum[1]
path_to_aic_bd_proj = path_to__00W5@p_00W5_paic_bd_proj
path_to_gamma_b = path_to__00W5@p_00W5_pgamma_b

# _00W8 einsum_eval_p_00W9_pM_mat
_00W8_temp_einsum = _00W8_partial_func(M_mat, _00W7)
p_00W9_p_00W7 = _00W8_temp_einsum[0]
p_00W9_pM_mat = _00W8_temp_einsum[1]
path_to__00W7 = path_to__00W9@p_00W9_p_00W7
path_to_M_mat = path_to__00W9@p_00W9_pM_mat

# _00W6 reshape_eval_p_00W7_pwing_gamma_w
path_to_wing_gamma_w = path_to__00W7@p_00W7_pwing_gamma_w
dy_dgamma_b = path_to_gamma_b.copy()
dy_daic_bd_proj = path_to_aic_bd_proj.copy()
dy_dM_mat = path_to_M_mat.copy()
dy_dwing_gamma_w = path_to_wing_gamma_w.copy()
dy_db = path_to_b.copy()