import numpy as np

rho = 997
# s = 10e-2 # 10 cm
s = 3# 10 cm
omg = np.deg2rad(10)

# l = 0.3 # meter
l = 2 # meter
t = np.zeros(20)
t[:10] = np.linspace(0,10,1)
little_w = omg*l
# capital_w = omg*l*np.cos(omg*t+alpha_0)
capital_w = little_w

m = 1/4*np.pi*s**2*rho
dx_da=1
T = m*little_w*capital_w - m*little_w**2/2*dx_da

CD_0 = 0.1936
CD_1 = 0.1412
alpha_deg = 0
alpha = alpha_deg / 180 * np.pi
CD_v = CD_0 + CD_1 * alpha**2
A = 2*3*2
v = 2
D = 0.5*rho*v**2*A*CD_0