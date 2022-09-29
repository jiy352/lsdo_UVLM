import time

import matplotlib.pyplot as plt
import openmdao.api as om
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_system_free import ODESystemModel
from lsdo_uvlm.examples.simple_wing_constant_aoa_sls_outputs.plunging_profile_outputs import ProfileSystemModel

from ozone.api import ODEProblem
import csdl

import numpy as np
from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show


from lsdo_uvlm.uvlm_preprocessing.actuation_model_temp import ActuationModel
from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh


x_20 = [0.13106796116504854, 0.13106796116504854, 0.15291262135922332, 0.2184466019417476, 0.3713592233009709, 0.48058252427184467, 0.6553398058252428, 0.8519417475728156, 1.1140776699029127, 1.441747572815534, 1.8786407766990292, 2.2936893203883497, 2.7087378640776687, 3.16747572815534, 3.626213592233011, 4.08495145631068, 4.609223300970874, 5.155339805825243, 5.6796116504854375, 6.203883495145631, 6.728155339805825, 7.252427184466019, 7.711165048543689, 8.148058252427184, 8.912621359223301, 8.563106796116505, 9.196601941747574, 9.196601941747574]
y_20 = [0.3557692307692308, 0.34401709401709407, 0.33440170940170943, 0.3290598290598291, 0.3322649572649573, 0.3386752136752137, 0.3461538461538462, 0.35683760683760685, 0.3685897435897436, 0.38141025641025644, 0.39743589743589747, 0.40811965811965817, 0.4166666666666667, 0.42521367521367526, 0.43162393162393164, 0.4380341880341881, 0.44230769230769235, 0.44764957264957267, 0.4508547008547009, 0.45299145299145305, 0.4572649572649573, 0.45940170940170943, 0.46153846153846156, 0.46260683760683763, 0.4647435897435898, 0.46367521367521375, 0.4658119658119659, 0.4658119658119659]



x_12 = [0.10465116279069767, 0.12209302325581395, 0.12209302325581395, 0.1569767441860465, 0.2616279069767442, 0.4011627906976744, 0.5581395348837209, 0.6802325581395349, 0.8197674418604651, 0.9767441860465116, 1.1162790697674418, 1.2732558139534884, 1.4127906976744187, 1.5523255813953487, 1.6744186046511627, 1.8313953488372092, 2.0406976744186047, 2.25, 2.4244186046511627, 2.6162790697674416, 2.808139534883721, 3.0174418604651163, 3.2093023255813953, 3.453488372093023, 3.7325581395348837, 4.02906976744186, 4.3604651162790695, 4.691860465116279, 5.040697674418604, 5.3895348837209305, 5.825581395348837, 6.2441860465116275, 6.697674418604651, 7.063953488372093, 7.482558139534883, 7.988372093023256, 8.441860465116278, 8.790697674418604, 9.05232558139535, 9.313953488372093]
y_12 = [0.3409863945578231, 0.33503401360544216, 0.3273809523809524, 0.320578231292517, 0.320578231292517, 0.3239795918367347, 0.3324829931972789, 0.3409863945578231, 0.3477891156462585, 0.35374149659863946, 0.35799319727891155, 0.3639455782312925, 0.369047619047619, 0.3724489795918367, 0.37670068027210885, 0.38095238095238093, 0.3852040816326531, 0.3886054421768707, 0.3920068027210884, 0.39710884353741494, 0.4013605442176871, 0.4039115646258503, 0.4064625850340136, 0.40901360544217685, 0.41241496598639454, 0.41581632653061223, 0.4183673469387755, 0.42091836734693877, 0.423469387755102, 0.4260204081632653, 0.42772108843537415, 0.429421768707483, 0.43112244897959184, 0.4328231292517007, 0.43452380952380953, 0.43452380952380953, 0.4362244897959183, 0.4370748299319728, 0.4370748299319728, 0.4370748299319728]


x_8 = [0.1092233009708738, 0.13106796116504854, 0.13106796116504854, 0.15291262135922332, 0.28398058252427183, 0.4587378640776699, 0.6553398058252428, 0.8737864077669903, 1.1359223300970873, 1.3980582524271845, 1.6820388349514563, 1.987864077669903, 2.2936893203883497, 2.599514563106795, 2.9271844660194164, 3.2111650485436893, 3.582524271844659, 3.9101941747572817, 4.325242718446602, 4.718446601941748, 5.111650485436893, 5.548543689320389, 5.963592233009709, 6.33495145631068, 6.902912621359223, 7.449029126213593, 7.995145631067961, 8.563106796116505, 8.912621359223301, 9.240291262135923]
y_8 = [0.3525641025641026, 0.33974358974358976, 0.32585470085470086, 0.311965811965812, 0.30662393162393164, 0.311965811965812, 0.32051282051282054, 0.3290598290598291, 0.33760683760683763, 0.3461538461538462, 0.35363247863247865, 0.36111111111111116, 0.3664529914529915, 0.3717948717948718, 0.3760683760683761, 0.3803418803418804, 0.3824786324786325, 0.3856837606837607, 0.38888888888888895, 0.3910256410256411, 0.3931623931623932, 0.39529914529914534, 0.39529914529914534, 0.39743589743589747, 0.39743589743589747, 0.39850427350427353, 0.3995726495726496, 0.3995726495726496, 0.40064102564102566, 0.40064102564102566]


x_4 = [0.1092233009708738, 0.08737864077669903, 0.08737864077669903, 0.06553398058252427, 0.06553398058252427, 0.1092233009708738, 0.1092233009708738, 0.1092233009708738, 0.1092233009708738, 0.13106796116504854, 0.30582524271844663, 0.5242718446601942, 0.7864077669902912, 1.0703883495145632, 1.354368932038835, 1.7038834951456312, 2.097087378640778, 2.490291262135921, 2.883495145631068, 3.2548543689320377, 3.6917475728155327, 4.150485436893204, 4.587378640776699, 5.024271844660194, 5.461165048543689, 5.8543689320388355, 6.247572815533981, 6.640776699029127, 7.077669902912621, 7.514563106796117, 7.842233009708738, 8.300970873786408, 8.781553398058252, 9.257767056955874]
y_4 = [0.3728632478632479, 0.39529914529914534, 0.43910256410256415, 0.5438034188034189, 0.47329059829059833, 0.3525641025641026, 0.3301282051282052, 0.3130341880341881, 0.29273504273504275, 0.280982905982906, 0.27029914529914534, 0.27564102564102566, 0.2820512820512821, 0.2873931623931624, 0.29273504273504275, 0.29807692307692313, 0.30341880341880345, 0.30662393162393164, 0.3087606837606838, 0.3108974358974359, 0.3130341880341881, 0.31410256410256415, 0.3151709401709402, 0.3162393162393163, 0.3162393162393163, 0.31730769230769235, 0.31730769230769235, 0.31730769230769235, 0.31730769230769235, 0.3183760683760684, 0.3183760683760684, 0.3183760683760684, 0.3183760683760684, 0.3183760683760684]


# cl_4 = np.loadtxt('cl4')
# cl_8 = np.loadtxt('cl8')
# cl_12 = np.loadtxt('cl12')
cl_20= np.loadtxt('cl20_full')
x = np.arange(32)*1/4
x = np.arange(1,160)*1/16



# plt.plot(x,cl_4,'b.')
# plt.plot(x,cl_8,'r.')
# plt.plot(x,cl_12,'g.')
plt.plot(x,cl_20[1:],'y.')


# plt.plot(x_4,y_4,'b')
# plt.plot(x_8,y_8,'r')
# plt.plot(x_12,y_12,'g')
plt.plot(x_20,y_20,'y')


# plt.legend(['AR=4','AR=8','AR=12','AR=20',])
# plt.legend(['AR=4','AR=8','AR=12','AR=20',])
plt.show()