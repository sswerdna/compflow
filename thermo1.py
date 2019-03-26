# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:06:15 2019

@author: andres11
"""

import numpy as np
import matplotlib.pyplot as plt

gamma_air = 1.4
gamma_methane = 1.32

vr = np.linspace(.5,1,300)

pr_isen_air = np.power(vr,-gamma_air)
pr_isen_methane = np.power(vr,-gamma_methane)
pr_hugon_air = ((gamma_air + 1)/(gamma_air - 1)*np.power(vr,-1) - 1)/((gamma_air + 1)/(gamma_air - 1)-np.power(vr,-1))
pr_hugon_methane = ((gamma_methane + 1)/(gamma_methane - 1)*np.power(vr,-1) - 1)/((gamma_methane + 1)/(gamma_methane - 1)-np.power(vr,-1))

plt.plot(pr_isen_air,vr,pr_isen_methane,vr,pr_hugon_air,vr,pr_hugon_methane,vr)
plt.legend(["Air - Isentropic","Methane - Isentropic","Air - Hugoniot", "Methane - Hugoniot"])
plt.xlabel("p2/p1")
plt.ylabel("v2/v1")