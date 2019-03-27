# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:49:40 2019

@author: andres11
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def total(M, gamma=1.4, R=287.05, T=None,p=None,rho=None,prnt=False):
    Tr = 1 + (gamma - 1)/2*np.power(M,2)
    pr = np.power(Tr,gamma/(gamma - 1))
    rho_r = np.power(Tr,1/(gamma-1))
    if prnt:
        print("T0/T=h0/h=%0.4f" % Tr)
        print("p0/p=%0.4f" % pr)
        print(u"\u03C10/\u03C1=%0.4f" % rho_r)
        print()
    if T is not None:
        T0 = Tr*T
        if prnt:print("T0=%0.1f" % T0)
    if p is not None:
        p0 = pr*p
        if prnt:print("p0=%0.2f" % p0)
    if rho is not None:
        rho_0 = rho_r*rho
        if prnt:print("\u03C1=%0.2f" % rho_0)
    return np.array([Tr,pr,rho_r])

def normal_shock(M1,gamma=1.4,R=287.05,prnt=False):
    M2 = np.power((2+(gamma-1)*np.power(M1,2))/(2*gamma*np.power(M1,2)-(gamma-1)),0.5)
    pr = 1 + 2 * gamma/(gamma + 1) * (np.power(M1,2)-1)
    rho_r = (gamma + 1)* np.power(M1,2)/(2 + (gamma - 1)*np.power(M1,2))
    Tr = pr/rho_r
    if prnt:
        print("M2 = %0.3f" % M2)
        print("T2/T1 = %0.4f" % Tr)
        print("p2/p1 = %0.4f" % pr)
        print ("\u03C12/\u03C11 = %0.4f" % rho_r)
    return np.array([M2,Tr,pr,rho_r])
    
def oblique(M1, beta=None,theta=None,gamma=1.4,R=287.05,prnt=False):
    #angles given in degrees
    corner_x1 = np.linspace(-1,0,100)
    corner_x2 = np.linspace(0,2,100)
    corner_y1 = np.zeros(corner_x1.shape)
        
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
        if prnt:
            print("\u03B8=%0.1f\u00B0" % (theta*180/np.pi))
            print(Mn2, Tr, pr, rho_r)
            print("M2=%0.2f" % M2)
            shock_y = np.tan(beta_rad)*corner_x2
            corner_y2 = np.tan(theta)*corner_x2
            plt.plot(corner_x1,corner_y1,"k",corner_x2,corner_y2,"k",corner_x2,shock_y,"b")
            plt.ylim(-0.1,2.9)
        return [Mn2,Tr,pr,rho_r, theta]
        
    if beta is None:
        #solve beta-theta-M for beta
        #there will be a weak and a strong solution, but the weak solution
        #is generally favored
        
        btm = lambda b:np.tan(theta_rad) -  2 * np.power(np.tan(b),-1) * (np.power(M1*np.sin(b),2)-1) \
        / (2 + np.power(M1,2)*(gamma + np.cos(2*b)))
        beta_w = opt.newton(btm,np.pi/4)
        beta_s = opt.newton(btm,np.pi/2)
        if prnt:
            print("Weak solution: \u03B2=%0.1f\u00B0" % (beta_w*180/np.pi))
            print("Strong Solution: \u03B2=%0.1f\u00B0" % (beta_s*180/np.pi))
        
        Mn1_w = M1*np.sin(beta_w)
        Mn1_s = M1*np.sin(beta_s)
        Mn2_w, Tr_w, pr_w, rho_r_w = normal_shock(Mn1_w,gamma=gamma,R=R)
        Mn2_s, Tr_s, pr_s, rho_r_s = normal_shock(Mn1_s,gamma=gamma,R=R)
        M2_w = Mn2_w/np.sin(beta_w-theta_rad)
        M2_s = Mn2_s/np.sin(beta_s-theta_rad)
        if prnt:
            print("M2 (Weak Solution) = %0.2f" % M2_w)
            print("M2 (Strong Solution) = %0.2f" % M2_s)
            shock_y_w = np.tan(beta_w)*corner_x2
            shock_y_s = np.tan(beta_s)*corner_x2
            corner_y2 = np.tan(theta_rad)*corner_x2
            plt.plot(corner_x1,corner_y1,"k",corner_x2,corner_y2,"k",corner_x2,shock_y_w,"b",corner_x2,shock_y_s,"b--")
            plt.ylim(-0.1,2.9)
        return [[M2_w,Tr_w,pr_w,rho_r_w, beta_w*180/np.pi],[M2_s,Tr_s,pr_s,rho_r_s,beta_s*180/np.pi]]

def area_ratio(M,gamma=1.4,R=287.05):
    area_r = 1.0/M * np.power(2/(gamma + 1)*(1 + 0.5 * (gamma - 1)*np.power(M,2.0)),(gamma+1)/(2 * (gamma-1)))
    return area_r

def prandtl_meyer(M,gamma=1.4,R=287.05):
    return np.power((gamma+1)/(gamma-1),0.5)*np.arctan(np.power((gamma-1)/(gamma+1)*(np.power(M,2)-1),0.5))- \
            np.arctan(np.power(np.power(M,2)-1,0.5))

def de_laval_nozzle(inlet_press, backpressure, exit_area, throat_area=1, gamma = 1.4, R = 287.05):
    area_r = exit_area/throat_area
    press_isen = lambda M: total(M,gamma=gamma,R=R)[2]-inlet_press/backpressure
    isen_mach = opt.newton(press_isen,1)
    print(isen_mach)
    area_rat_f = lambda M:area_ratio(M,gamma=gamma,R=R)-area_r
    area_rat_sub = opt.newton(area_rat_f,0.1)
    area_rat_super = opt.newton(area_rat_f,1.5)
    print(area_rat_sub)
    print(area_rat_super)
    
def intersecting_shock(M1, theta_top,theta_bottom):
    shock_1 = oblique(M1,theta=theta_top)
    shock_2 = oblique(M1,theta=theta_bottom)
    theta_1_prime = lambda phi: theta_top+phi
    theta_2_prime = lambda phi: theta_bottom-phi
    p1 = shock_1[0][2]
    p2 = shock_2[0][2]
    M1 = shock_1[0][0]
    M2 = shock_2[0][0]
    press_diff = lambda phi: p1*oblique(M1,theta = theta_1_prime(phi))[0][2] - p2*oblique(M2,theta = theta_2_prime(phi))[0][2]
    phi_actual = opt.newton(press_diff,10)
    print(phi_actual)
    