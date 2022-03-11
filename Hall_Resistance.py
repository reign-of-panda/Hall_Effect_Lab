# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:22:09 2022

@author: therm
"""

# Hall resistance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12.5
from scipy.optimize import curve_fit
import math

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def linear(x, m, c):
    return m * x + c

def carrier_density(grad, grad_err, t, t_err=0):
    e = 1.6e-19
    density = abs(1/(grad * e * t))
    
    rel_grad_err = grad_err / grad
    rel_t_err = t_err / t
    tot_rel_err = rel_grad_err + rel_t_err
    
    density_err = density * tot_rel_err    
    
    return density, density_err
    

dF1 = pd.read_excel('N_Type_GaS_Hall_Resistance.xlsx')
t1 = 3e-6

mfs = np.array(dF1['Magnetic Field Strength (mT)']) / 1000
hall_res = np.array(dF1['Hall_Res (ohm)'])
mfs_err = np.array(dF1['MFS Err (mT)']) / 1000
hall_res_err = np.array(dF1['Hall_Res_err (ohm)'])

par1, cov1 = curve_fit(linear, mfs, hall_res, sigma = hall_res_err, absolute_sigma = True)
fit1 = linear(mfs, *par1)

grad = truncate(par1[0], 3)
grad_err = truncate(cov1[0][0] ** 0.5, 5)
interp = truncate(par1[1], 3)
interp_err = truncate(cov1[1][1] ** 0.5, 5)

rho, rho_err = carrier_density(par1[0], grad_err, t1)

carrier_density = truncate(rho, 3)
car_den_err = truncate(rho_err, 3)

print("Gradient:", grad, grad_err)
print("Intercept:", interp, interp_err)
print("Carrier Density:", carrier_density, car_den_err)

plt.errorbar(mfs, hall_res, yerr = hall_res_err, xerr = mfs_err, capsize = 2, fmt = 'o', label = 'N-Type GaS Data')
plt.plot(mfs, fit1, label = "Fit")
text_x = -0.04
text_y = -1
plt.text(text_x, text_y, "Gradient: {} $\pm$ {}".format(grad, grad_err), fontsize = 12)
plt.text(text_x, text_y - 0.3, "Intercept: {} $\pm$ {}".format(interp, interp_err), fontsize = 12)
# plt.text(0.1, -1, "Carrier Density: {}".format(carrier_density), fontsize = 12)

plt.minorticks_on()
plt.xlabel("Magnetic Field Strength (T)")
plt.ylabel("Hall Resistance ($\Omega$)")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.legend()
plt.show()

#%%
"""
Now for P-Type GaS
"""





