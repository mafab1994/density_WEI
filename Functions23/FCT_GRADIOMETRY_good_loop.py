# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:48:17 2021

@author: s2110831
"""
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as linalg

from FCT_laplacian import *


def GRAD_cst(dist_r, U_filter, nt_sub, n_rx, n_rz, dt_sub,order):
    
    Uxx_GRAD_CST  = np.empty((n_rx, n_rz, nt_sub))
    Uzz_GRAD_CST  = np.empty((n_rx, n_rz, nt_sub))
    Utt_GRAD_CST  = np.empty((n_rx, n_rz, nt_sub))
    
    for i in range(1,nt_sub-1):
        
        if order ==2:
        

            for l in range(1,n_rx-1):
                Uxx_GRAD_CST[l, :,i] = (U_filter[l+1,:,i] - (2*U_filter[l,:,i]) + U_filter[l-1,:,i])/dist_r**2
            for k in range(1,n_rz-1):
                Uzz_GRAD_CST[:,k,i] = (U_filter[:,k+1,i] - (2*U_filter[:,k,i]) +U_filter[:,k-1,i])/dist_r**2
        else:
        
            
            for l in range(2,n_rx-2):
                Uxx_GRAD_CST[l, :,i] = (-1/12*U_filter[l+2,:,i] + 4/3*U_filter[l+1,:,i] - (5/2*U_filter[l,:,i]) + 4/3*U_filter[l-1,:,i] - 1/12*U_filter[l-2,:,i])/ dist_r**2 #nominator*2
            for k in range (2,n_rz-2):
                Uzz_GRAD_CST[:,k,i] = (-1/12*U_filter[:,k+2,i] + 4/3*U_filter[:,k+1,i] - (5/2*U_filter[:,k,i]) + 4/3*U_filter[:,k-1,i] - 1/12*U_filter[:,k-2,i])/ dist_r**2

        # Time derivative, second order accurate
        Utt_GRAD_CST[:,:,i] = (U_filter[:,:,i-1] - (2* U_filter[:,:,i]) + U_filter[:,:,i+1])/(dt_sub**2)
        
    return Uxx_GRAD_CST, Uzz_GRAD_CST, Utt_GRAD_CST


def GRAD_var(dist_r, U_filter, nt_sub, n_rx, n_rz, dt_sub, rho_m_sub):
    
    Uxx_GRAD_VAR  = np.empty((n_rx, n_rz, nt_sub))
    Uzz_GRAD_VAR  = np.empty((n_rx, n_rz, nt_sub))
    Utt_GRAD_VAR  = np.empty((n_rx, n_rz, nt_sub))
    
    for i in range(1,nt_sub-1):
    
        for l in range(1,n_rx-1):
            Uxx_GRAD_VAR[l, :,i] =  (1 / (2 * (dist_r**2) )) *  (( (U_filter[l+1,:,i] - U_filter[l,:,i]) * (rho_m_sub[l+1,:] + rho_m_sub[l,:] ) / rho_m_sub[l+1,:] ) - ( (U_filter[l,:,i] - U_filter[l-1,:,i]) * (rho_m_sub[l,:] + rho_m_sub[l-1,:] ) / rho_m_sub[l-1,:] ) )
        for k in range(1,n_rz-1):
            Uzz_GRAD_VAR[:,k,i] = (1 / (2 * (dist_r**2) )) *  (( (U_filter[:,k+1,i] - U_filter[:,k,i]) * (rho_m_sub[:,k+1] + rho_m_sub[:,k] ) / rho_m_sub[:,k+1] ) - ( (U_filter[:,k,i] - U_filter[:,k-1,i]) * (rho_m_sub[:,k] + rho_m_sub[:,k-1] ) / rho_m_sub[:,k-1] ) )
        
        # Time derivative, second order accurate
        Utt_GRAD_VAR[:,:,i] = (U_filter[:,:,i-1] - (2* U_filter[:,:,i]) + U_filter[:,:,i+1])/(dt_sub**2)

        
    return Uxx_GRAD_VAR, Uzz_GRAD_VAR, Utt_GRAD_VAR


def GRAD_var_ELASTIC(dist_r, U_filter, nt_sub, n_rx, n_rz, dt_sub, rho_m_sub):
    
    Uxx_GRAD_VAR  = np.empty((n_rx, n_rz, nt_sub))
    Uzz_GRAD_VAR  = np.empty((n_rx, n_rz, nt_sub))
    Utt_GRAD_VAR  = np.empty((n_rx, n_rz, nt_sub))
    
    for i in range(1,nt_sub-1):
    
        for l in range(1,n_rx-1):
            Uxx_GRAD_VAR[l, :,i] =  (1 / (2 * (dist_r**2) )) *  (( (U_filter[l+1,:,i] - U_filter[l,:,i]) * (rho_m_sub[l-1,:] + rho_m_sub[l,:] ) / rho_m_sub[l-1,:] ) - ( (U_filter[l,:,i] - U_filter[l-1,:,i]) * (rho_m_sub[l,:] + rho_m_sub[l+1,:] ) / rho_m_sub[l+1,:] ) )
        for k in range(1,n_rz-1):
            Uzz_GRAD_VAR[:,k,i] = (1 / (2 * (dist_r**2) )) *  (( (U_filter[:,k+1,i] - U_filter[:,k,i]) * (rho_m_sub[:,k-1] + rho_m_sub[:,k] ) / rho_m_sub[:,k-1] ) - ( (U_filter[:,k,i] - U_filter[:,k-1,i]) * (rho_m_sub[:,k] + rho_m_sub[:,k+1] ) / rho_m_sub[:,k+1] ) )
        
        # Time derivative, second order accurate
        Utt_GRAD_VAR[:,:,i] = (U_filter[:,:,i-1] - (2* U_filter[:,:,i]) + U_filter[:,:,i+1])/(dt_sub**2)

        
    return Uxx_GRAD_VAR, Uzz_GRAD_VAR, Utt_GRAD_VAR


def GRAD_var3D(dist_r, U_filter, nt_sub, n_rx,n_ry, n_rz, dt_sub, rho_m_sub):
    
    Uxx_GRAD_VAR  = np.ones((n_rx,n_ry, n_rz, nt_sub))*1e-20
    Uyy_GRAD_VAR  = np.ones((n_rx,n_ry, n_rz, nt_sub))*1e-20
    Uzz_GRAD_VAR  = np.ones((n_rx,n_ry, n_rz, nt_sub))*1e-20
    Utt_GRAD_VAR  = np.ones((n_rx,n_ry, n_rz, nt_sub))*1e-20
    
    for i in range(1,nt_sub-1):
    
        for l in range(1,n_rx-1):
            Uxx_GRAD_VAR[l,:,:,i] =  (1 / (2 * (dist_r**2) )) *  (( (U_filter[l+1,:,:,i] - U_filter[l,:,:,i]) * (rho_m_sub[l+1,:,:] + rho_m_sub[l,:,:] ) / rho_m_sub[l+1,:,:] ) - ( (U_filter[l,:,:,i] - U_filter[l-1,:,:,i]) * (rho_m_sub[l,:,:] + rho_m_sub[l-1,:,:] ) / rho_m_sub[l-1,:,:] ) )
       
        for k in range(1,n_ry-1):
            Uyy_GRAD_VAR[:,k,:,i] = (1 / (2 * (dist_r**2) )) *  (( (U_filter[:,k+1,:,i] - U_filter[:,k,:,i]) * (rho_m_sub[:,k+1,:] + rho_m_sub[:,k,:] ) / rho_m_sub[:,k+1,:] ) - ( (U_filter[:,k,:,i] - U_filter[:,k-1,:,i]) * (rho_m_sub[:,k,:] + rho_m_sub[:,k-1,:] ) / rho_m_sub[:,k-1,:] ) )
        
        for mm in range(1,n_rz-1):
            Uzz_GRAD_VAR[:,:,mm,i] = (1 / (2 * (dist_r**2) )) *  (( (U_filter[:,:,mm+1,i] - U_filter[:,:,mm,i]) * (rho_m_sub[:,:,mm+1] + rho_m_sub[:,:,mm] ) / rho_m_sub[:,:,mm+1] ) - ( (U_filter[:,:,mm,i] - U_filter[:,:,mm-1,i]) * (rho_m_sub[:,:,mm] + rho_m_sub[:,:,mm-1] ) / rho_m_sub[:,:,mm-1] ) )
        
        # Time derivative, second order accurate
        Utt_GRAD_VAR[:,:,:,i] = (U_filter[:,:,:,i-1] - (2* U_filter[:,:,:,i]) + U_filter[:,:,:,i+1])/(dt_sub**2)
    
    Ugrad = Uxx_GRAD_VAR + Uyy_GRAD_VAR + Uzz_GRAD_VAR
        
    return Ugrad, Utt_GRAD_VAR


def GRAD_LAPLACE_CST(dist_r, U_filter, nt_sub, n_rx, n_rz, dt_sub, order,f_filt_cent):
    
    U_LAPLACE_CST      = np.zeros((n_rx, n_rz, nt_sub, len(f_filt_cent)))
    Utt_LAPLACE_CST    = np.zeros((n_rx, n_rz, nt_sub, len(f_filt_cent)))
    
    if order == 2:
        Dxx                = LaplacianMatrix_CST(n_rx, n_rz, dist_r, dist_r)#.todense()
        
    else:
        Dxx                = LaplacianMatrix_CST_4th(n_rx, n_rz, dist_r, dist_r)#.todense()
    for kk in range(0,len(f_filt_cent)):
        for i in range(1,nt_sub-1):

            # Space derivative, second order accurate
            prod = np.dot(Dxx, U_filter[:,:,i,kk].flatten())
            U_LAPLACE_CST[:,:,i,kk] = np.reshape(prod, (n_rx,n_rz))

            # Time derivative, second order accurate
            Utt_LAPLACE_CST[:,:,i,kk] = (U_filter[:,:,i-1,kk] - (2* U_filter[:,:,i,kk]) +\
                                         U_filter[:,:,i+1,kk])/(dt_sub**2)

        
    return U_LAPLACE_CST,  Utt_LAPLACE_CST, prod, Dxx


def GRAD_LAPLACE_CST_COMPLEX(dist_r, U_filter, nt_sub, n_rx, n_rz, dt_sub, order,f_filt_cent):
    
    U_LAPLACE_CST      = np.zeros((n_rx, n_rz, nt_sub, len(f_filt_cent)),dtype = 'complex_')
    Utt_LAPLACE_CST    = np.zeros((n_rx, n_rz, nt_sub, len(f_filt_cent)),dtype = 'complex_')
    
    if order == 2:
        Dxx                = LaplacianMatrix_CST(n_rx, n_rz, dist_r, dist_r)#.todense()
        
    else:
        Dxx                = LaplacianMatrix_CST_4th(n_rx, n_rz, dist_r, dist_r)#.todense()
    for kk in range(0,len(f_filt_cent)):
        for i in range(1,nt_sub-1):

            # Space derivative, second order accurate
            prod = np.dot(Dxx, U_filter[:,:,i,kk].flatten())
            U_LAPLACE_CST[:,:,i,kk] = np.reshape(prod, (n_rx,n_rz))

            # Time derivative, second order accurate
            Utt_LAPLACE_CST[:,:,i,kk] = (U_filter[:,:,i-1,kk] - (2* U_filter[:,:,i,kk]) +\
                                         U_filter[:,:,i+1,kk])/(dt_sub**2)

        
    return U_LAPLACE_CST,  Utt_LAPLACE_CST, prod, Dxx


def GRAD_LAPLACE_VAR(dist_r, U_filter, nt_sub, n_rx, n_rz, dt_sub, rho_m_sub,f_filt_cent):
    
    U_LAPLACE_VAR      = np.empty((n_rx, n_rz, nt_sub, len(f_filt_cent)))
    Utt_LAPLACE_VAR    = np.empty((n_rx, n_rz, nt_sub, len(f_filt_cent)))
    Dxx                = LaplacianMatrix_VAR(n_rx, n_rz, dist_r, dist_r, rho_m_sub)#.todense()
    for kk in range(0,len(f_filt_cent)):
        for i in range(1,nt_sub-1):

            # Space derivative, second order accurate
            prod = np.dot(Dxx, U_filter[:,:,i,kk].flatten())
            U_LAPLACE_VAR[:,:,i,kk] = np.reshape(prod, (n_rx,n_rz))

            # Time derivative, second order accurate
            Utt_LAPLACE_VAR[:,:,i,kk] = (U_filter[:,:,i-1,kk] - (2* U_filter[:,:,i,kk]) +\
                                         U_filter[:,:,i+1,kk])/(dt_sub**2)

        
    return U_LAPLACE_VAR ,  Utt_LAPLACE_VAR , prod, Dxx
