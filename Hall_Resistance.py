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

def mobility(n,rho):
    e = 1.6e-19
    return 1 / (e * n * rho)

resistivity = [5.35e-5, 2.91e-4, 8.229e-5]
    

dF1 = pd.read_excel('N_Type_GaS_Hall_Resistance.xlsx')
t1 = 3e-6

mfs = np.array(dF1['Magnetic Field Strength (mT)']) / 1000
hall_res = np.array(dF1['Hall_Res (ohm)']) *-1
mfs_err = np.array(dF1['MFS Err (mT)']) / 1000 
hall_res_err = np.array(dF1['Hall_Res_err (ohm)'])

par1, cov1 = curve_fit(linear, mfs, hall_res, sigma = hall_res_err, absolute_sigma = True)
fit1 = linear(mfs, *par1)

grad = truncate(par1[0], 3)
grad_err = truncate(cov1[0][0] ** 0.5, 5)
interp = truncate(par1[1], 3)
interp_err = truncate(cov1[1][1] ** 0.5, 5)

rho, rho_err = carrier_density(par1[0], grad_err, t1)

car_den = truncate(rho, 3)
car_den_err = truncate(rho_err, 3)

print("Gradient:", grad, grad_err)
print("Intercept:", interp, interp_err)
print("Carrier Density:", car_den, car_den_err)

plt.errorbar(mfs, hall_res, yerr = hall_res_err, xerr = mfs_err, capsize = 2, fmt = 'o', label = 'N-Type GaAs Data')
plt.plot(mfs, fit1, label = "Fit")
text_x = -0.3
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
plt.savefig('N_type_GaAs_HallPlot',dpi = 300, bbox_inches="tight")
plt.show()


#%%
"""
Now for P-Type GaS
"""

dF2 = pd.read_excel('P_Type_GaS_Hall_Resistance.xlsx')
t2 = 2.7e-6

mfs = np.array(dF2['Magnetic Field Strength (mT)']) / 1000
hall_res_P = np.array(dF2['Hall_Res (ohm)'])*-1
mfs_err_P = np.array(dF2['MFS Err (mT)']) / 1000
hall_res_err_P = np.array(dF2['Hall_Res_err (ohm)'])

par2, cov2 = curve_fit(linear, mfs, hall_res_P, sigma = hall_res_err_P, absolute_sigma = True)
fit2 = linear(mfs, *par2)

grad_P = truncate(par2[0], 3)
grad_err_P = truncate(cov2[0][0] ** 0.5, 5)
interp_P = truncate(par2[1], 3)
interp_err_P = truncate(cov2[1][1] ** 0.5, 5)

rho_P, rho_err_P = carrier_density(par2[0], grad_err_P, t2)

carrier_density_P = truncate(rho_P, 3)
car_den_err_P = truncate(rho_err_P, 3)

print("Gradient:", grad_P, grad_err_P)
print("Intercept:", interp_P, interp_err_P)
print("Carrier Density:", carrier_density_P, car_den_err_P)

plt.errorbar(mfs, hall_res_P, yerr = hall_res_err_P, xerr = mfs_err_P, capsize = 2, fmt = 'o', label = 'P-Type GaAs Data')
plt.plot(mfs, fit2, label = "Fit")
text_x = -0.04
text_y = -0.3
plt.text(text_x, text_y, "Gradient: {} $\pm$ {}".format(grad_P, grad_err_P), fontsize = 12)
plt.text(text_x, text_y - 0.1, "Intercept: {} $\pm$ {}".format(interp_P, interp_err_P), fontsize = 12)
# plt.text(0.1, -1, "Carrier Density: {}".format(carrier_density), fontsize = 12)

plt.minorticks_on()
plt.xlabel("Magnetic Field Strength (T)")
plt.ylabel("Hall Resistance ($\Omega$)")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.legend()
plt.savefig('P_type_GaAs_HallPlot',dpi = 300, bbox_inches="tight")
plt.show()

#%%
"""
Now for InSb
"""

dF3 = pd.read_excel('InSb_Hall_Resistance.xlsx')
t3 = 1e-6

mfs = np.array(dF3['Magnetic Field Strength (mT)']) / 1000
hall_res_InSb = np.array(dF3['Hall_Res (ohm)'])
mfs_err_InSb = np.array(dF3['MFS Err (mT)']) / 1000
hall_res_err_InSb = np.array(dF3['Hall_Res_err (ohm)']) * 100

par3, cov3 = curve_fit(linear, mfs, hall_res_InSb, sigma = hall_res_err_InSb, absolute_sigma = True)
fit3 = linear(mfs, *par3)

grad_InSb = truncate(par3[0], 3)
grad_err_InSb = truncate(cov3[0][0] ** 0.5, 5)
interp_InSb = truncate(par3[1], 3)
interp_err_InSb = truncate(cov3[1][1] ** 0.5, 5)

rho_InSb, rho_err_InSb = carrier_density(par3[0], grad_err_InSb, t3)

carrier_density_InSb = truncate(rho_InSb, 3)
car_den_err_InSb = truncate(rho_err_InSb, 3)

print("Gradient:", grad_InSb, grad_err_InSb)
print("Intercept:", interp_InSb, interp_err_InSb)
print("Carrier Density:", carrier_density_InSb, car_den_err_InSb)

plt.errorbar(mfs, hall_res_InSb, yerr = hall_res_err_InSb, xerr = mfs_err_InSb, capsize = 2, fmt = 'o', label = 'InSb Data')
plt.plot(mfs, fit3, label = "Fit")
text_x = -0.3
text_y = -50
plt.text(text_x, text_y, "Gradient: {} $\pm$ {}".format(grad_InSb, grad_err_InSb), fontsize = 12)
plt.text(text_x, text_y - 15, "Intercept: {} $\pm$ {}".format(interp_InSb, interp_err_InSb), fontsize = 12)
# plt.text(0.1, -1, "Carrier Density: {}".format(carrier_density), fontsize = 12)

plt.minorticks_on()
plt.xlabel("Magnetic Field Strength (T)")
plt.ylabel("Hall Resistance ($\Omega$)")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.legend()
plt.savefig('InSb_HallPlot', dpi = 300, bbox_inches="tight")
plt.show()

#%%
"""
Calculating Mobilities
"""
print('Mobility N-type (cm^2 V^-1 s^-1):', mobility(car_den, resistivity[0])*10000)
print('Mobility P-type (cm^2 V^-1 s^-1):', mobility(carrier_density_P, resistivity[1])*10000)
print('Mobility InSb (cm^2 V^-1 s^-1):', mobility(carrier_density_InSb, resistivity[2])*10000)


#%%

def res(mfs, hall_res, y_err, fit):
    hres = 0
    hres = hall_res - fit
    
    plt.errorbar(mfs, hres, yerr = y_err, fmt = 'o', label = 'InSb Residuals', marker = '.', markersize = 4, c = 'black', capsize = 2)
    
    plt.minorticks_on()
    plt.xlabel("Magnetic Field Strength (T)")
    plt.ylabel("Hall Resistance Residuals ($\Omega$)")
    plt.grid(which = 'minor', alpha = 0.2)
    plt.grid(which = 'major')
    plt.legend()
    plt.savefig('InSb_HallPlot')
    plt.show()
    return hres


#%%

par4, cov4 = curve_fit(linear, mfs[4:9], hall_res_InSb[4:9], sigma = hall_res_err_InSb[4:9], absolute_sigma = True)
fit4 = linear(mfs, *par4)


grad_InSb = truncate(par4[0], 3)
grad_err_InSb = truncate(cov4[0][0] ** 0.5, 5)
interp_InSb = truncate(par4[1], 3)
interp_err_InSb = truncate(cov4[1][1] ** 0.5, 5)

rho_InSb, rho_err_InSb = carrier_density(par4[0], grad_err_InSb, t3)

carrier_density_InSb = truncate(rho_InSb, 3)
car_den_err_InSb = truncate(rho_err_InSb, 3)

print("Gradient:", grad_InSb, grad_err_InSb)
print("Intercept:", interp_InSb, interp_err_InSb)
print("Carrier Density:", carrier_density_InSb, car_den_err_InSb)

plt.errorbar(mfs, hall_res_InSb, yerr = hall_res_err_InSb, xerr = mfs_err_InSb, capsize = 2, fmt = 'o', label = 'InSb Data')
plt.plot(mfs, fit4, label = "Fit")
text_x = -0.3
text_y = -50
plt.text(text_x, text_y, "Gradient: {} $\pm$ {}".format(grad_InSb, grad_err_InSb), fontsize = 12)
plt.text(text_x, text_y - 15, "Intercept: {} $\pm$ {}".format(interp_InSb, interp_err_InSb), fontsize = 12)
# plt.text(0.1, -1, "Carrier Density: {}".format(carrier_density), fontsize = 12)

plt.minorticks_on()
plt.xlabel("Magnetic Field Strength (T)")
plt.ylabel("Hall Resistance ($\Omega$)")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.legend()
plt.show()

#%%

def cubic(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d
    

def cubic_fit(x,y, y_err, color_line, color_dot, save = False):
    guess = [0,0,0,0]
    par, cov = curve_fit(cubic, x, y, guess, absolute_sigma = True)
    fit = cubic(x, par[0], par[1], par[2], par[3])
    
    x_1 = np.linspace(min(x), max(x), 1000)
    y_1 = cubic(x_1, par[0], par[1], par[2], par[3])
    print(par)
    
    plt.plot(x_1,y_1, c = color_line)
    plt.errorbar(x, y, yerr = y_err, fmt = 'o', label = 'InSb Temperature Dependence',marker = '.', markersize = 4, c = color_dot, capsize = 2)
    plt.minorticks_on()
    
    # plt.xlabel("Magnetic Field Strength (T)")
    # plt.ylabel("Hall Resistance ($\Omega$)")
    plt.xlabel("Temperature ($^{o}$C)")
    plt.ylabel("Resistance ($\Omega$)")
    
    plt.grid(which = 'minor', alpha = 0.2)
    plt.grid(which = 'major')
    plt.legend()
    
    if save:
        plt.savefig('mag_res.png',dpi = 300, bbox_inches="tight") 
    
    plt.show()
    
#%%
"""
Temperature dependence 
"""

dF_temp = pd.read_excel('Indium_Temperature.xlsx')

temp = np.array(dF_temp['Temperature (C)'])
resistance = np.array(dF_temp['Voltage (mV)'])
err = np.ones(len(temp)) * hall_res_err_InSb[0]

plt.errorbar(temp, resistance, yerr = err, fmt = 'o', label = '', marker = '.', markersize = 4, c = 'red', capsize = 2)

plt.minorticks_on()
plt.xlabel("Temperature ($^{o}$C)")
plt.ylabel("Hall Resistance ($\Omega$)")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.legend()
plt.show()

#%%

x = np.array([1,1.000000001])
y = np.array([0,1000])

plt.plot(x*23.4, y, linestyle = "--", c = 'black')
plt.plot(x*24.4, y, linestyle = "--", c = 'black')
plt.xlim([21.8, 26.4])
plt.ylim([27.35, 29.9])

plt.savefig('temp_dependence.png')














