# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:03:44 2022

@author: therm
"""

import numpy as np

def sheet_res(RA, RB, f_RA_RB = 1, RA_err = 0, RB_err = 0):
    """
    RA: The RA value
    RB: The RB value
    f_RA_RB: The correction factor
    """
    
    sheet_resistance = np.pi/np.log(2) * (RA + RB) / 2 * f_RA_RB
    error = (RA_err + RB_err) * np.pi/(2*np.log(2))
    return sheet_resistance, error

SR, err = sheet_res(18.32, 30.78, 0.97, 0.01, 0.01)

print(SR, err)

t = 2.7e-6
def resistivity(t, SR):
    return t * SR

rho = resistivity(t, SR)

print(rho)