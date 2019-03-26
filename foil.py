# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:14:57 2018

@author: andres11
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

NACA = "5508"

t = float(NACA[2:])/100
m = float(NACA[0])/100
p = float(NACA[1])/10
if p==0:
    p=0.5
    
plane = 'right'
def camber_line(x, m, p, c=1):
    leading_mask = np.less_equal(x, p).astype(int)
    trailing_mask = np.greater(x, p).astype(int)
    underlying = 2*p*x/c-np.power(x/c,2)
    leading_val = m/p**2*underlying
    trailing_val = m/(1-p)**2*(1-2*p+underlying)
    leading = leading_mask * leading_val
    trailing = trailing_mask * trailing_val
    return leading+trailing

def angle(x, m, p, c=1):
    leading_mask = np.less_equal(x, p).astype(int)
    trailing_mask = np.greater(x, p).astype(int)
    underlying = p-x/c
    leading_val = 2*m/p**2*underlying
    trailing_val = 2*m/(1-p**2)*underlying
    leading = leading_mask * leading_val
    trailing = trailing_mask * trailing_val
    return (np.cos(np.arctan(leading+trailing)),np.sin(np.arctan(leading+trailing)))

def thickness(x, t, zero_trailing_edge=True, c=1):
    if zero_trailing_edge:    
        thk = 5*t*(0.2969*np.power(x,0.5)-0.1260*x - 0.3516 * np.power(x,2) \
                   +0.2843*np.power(x,3) - 0.1036*np.power(x,4))
    else:
        thk = 5*t*(0.2969*np.power(x,0.5)-0.1260*x - 0.3516 * np.power(x,2) \
                   +0.2843*np.power(x,3) - 0.1015*np.power(x,4)) 
    return thk

xs = np.linspace(0,1,501)
cl = camber_line(xs, m, p)
ang_c, ang_s = angle(xs, m, p)
thk = thickness(xs, t)

xu = xs - thk*ang_s
xl = xs + thk*ang_s
yu = cl + thk*ang_c
yl = cl - thk*ang_c

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.set_ylim(-0.6,0.6)
ax.set_xlim(-0.1,1.1)
ax.plot(xu,yu,'#0F0F0F',xl,yl,'#0F0F0F')
fig.show()



with open(NACA+"_u.txt",'w') as fpu, open(NACA+"_l.txt",'w') as fpl:    
    cwu = csv.writer(fpu)
    cwl = csv.writer(fpl)
    for i in range(len(xu)):
        if plane.lower() == "front":
            cwu.writerow([xu[i],yu[i],0])
            cwl.writerow([xl[i],yl[i],0])
        if plane.lower() == "right":
            cwu.writerow([0,xu[i],yu[i]])
            cwl.writerow([0,xl[i],yl[i]])
        if plane.lower() == 'top':
            cwu.writerow([xu[i],0,yu[i]])
            cwl.writerow([xl[i],0,yl[i]])
        
            