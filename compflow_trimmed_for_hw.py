# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:49:40 2019
@author: andres11
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def normal_shock(M1,gamma=1.4,R=287.05):
    #Solves and returns normal shock ratios
    M2 = np.power((2+(gamma-1)*np.power(M1,2))/(2*gamma*np.power(M1,2)-(gamma-1)),0.5)
    pr = 1 + 2 * gamma/(gamma + 1) * (np.power(M1,2)-1)
    rho_r = (gamma + 1)* np.power(M1,2)/(2 + (gamma - 1)*np.power(M1,2))
    Tr = pr/rho_r
    return np.array([M2,Tr,pr,rho_r])
    
def oblique(M1, beta=None,theta=None,gamma=1.4,R=287.05,):
    #input given in degrees, so we need to convert to radians
    if beta is not None:
        beta_rad = beta*np.pi/180
    if theta is not None:
        theta_rad = theta*np.pi/180 
    
    if theta is None:
        #Solve beta-theta-M for theta
        #There is only one solution here
        Mn1 = M1*np.sin(beta_rad)
        tan_theta = 2 * np.power(np.tan(beta_rad),-1) * (np.power(M1*np.sin(beta_rad),2)-1) \
        / (2 + np.power(M1,2)*(gamma + np.cos(2*beta_rad)))
        theta = np.arctan(tan_theta)
        Mn2, Tr, pr, rho_r = normal_shock(Mn1, gamma=gamma,R=R)
        M2 = Mn2/np.sin(beta_rad-theta)
        return [M2,Tr,pr,rho_r, theta]
        
    if beta is None:
        #solve beta-theta-M for beta
        #there will be a weak and a strong solution, but the weak solution
        #is generally favored
        #helper function for solver
        btm = lambda b:np.tan(theta_rad) -  2 * np.power(np.tan(b),-1) * (np.power(M1*np.sin(b),2)-1) \
        / (2 + np.power(M1,2)*(gamma + np.cos(2*b)))
        # opt.newton solves for the zeros of the function starting with the second value.
        beta_w = opt.newton(btm,np.pi/4) 
        beta_s = opt.newton(btm,np.pi/2)
              
        Mn1_w = M1*np.sin(beta_w)
        Mn1_s = M1*np.sin(beta_s)
        Mn2_w, Tr_w, pr_w, rho_r_w = normal_shock(Mn1_w,gamma=gamma,R=R)
        Mn2_s, Tr_s, pr_s, rho_r_s = normal_shock(Mn1_s,gamma=gamma,R=R)
        M2_w = Mn2_w/np.sin(beta_w-theta_rad)
        M2_s = Mn2_s/np.sin(beta_s-theta_rad)
        return [[M2_w,Tr_w,pr_w,rho_r_w, beta_w*180/np.pi],[M2_s,Tr_s,pr_s,rho_r_s,beta_s*180/np.pi]]

    
def intersecting_shock(M1, theta_top,theta_bottom):
    #First we calculate properties after the first set of shocks
    shock_1 = oblique(M1,theta=theta_top)
    shock_2 = oblique(M1,theta=theta_bottom)
    p1 = shock_1[0][2]
    p2 = shock_2[0][2]
    M1 = shock_1[0][0]
    M2 = shock_2[0][0]
    #cross-check with hand calculations
    print(M1,M2,"\n",p1,p2)
    #these are helper functions which will be used during the optimization        
    theta_1_prime = lambda phi: theta_top+phi
    theta_2_prime = lambda phi: theta_bottom-phi
    press_diff = lambda phi: p1*oblique(M1,theta = theta_1_prime(phi))[0][2] - p2*oblique(M2,theta = theta_2_prime(phi))[0][2]
    phi_actual = opt.newton(press_diff,10)
    print(phi_actual)
    
    #compute final properties for verification
    theta_1_p=theta_1_prime(phi_actual)
    theta_2_p=theta_2_prime(phi_actual)
    shock_1_prime = oblique(M1,theta=theta_1_p)
    shock_2_prime = oblique(M2,theta=theta_2_p)
    press_top = shock_1_prime[0][2]*shock_1[0][2]
    press_bot = shock_2_prime[0][2]*shock_2[0][2]
    print(press_top,press_bot) # confirm equality