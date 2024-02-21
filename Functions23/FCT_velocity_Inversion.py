# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:06:30 2021

@author: s2110831
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from FCT_laplacian import *


# ## Via Inversion

def FCT_INV_(eps,n_rx, n_rz, lhs, rhs, nt_sub, dist_r):
           
    ##################################################################################################################################################
    #                               CONSTANT DENSITY MODEL 
    ##################################################################################################################################################
    
    # Initialization of variables 
    LHS      = np.zeros((n_rx**2, n_rz**2,nt_sub))
    RHS      = np.zeros((n_rx**2, n_rz**2,nt_sub))
    
    # Coefficient matrix
    # ex       = np.ones(n_rx)
    # data     = np.array([ex, -2 * ex, ex])
    # offsets  = np.array([-1, 0, 1])
    # Dxx      = scipy.sparse.dia_matrix((data, offsets), shape=(n_rx**2, n_rz**2)).toarray() * 1/dist_r**2
    # Dxx_c    = np.conj(Dxx)
    Dxx                 = LaplacianMatrix_CST(n_rx, n_rz, dist_r, dist_r)#.todense()
    Dxx_c    = np.matrix.getH(Dxx)
    
    
    
    # Objective function?
    #D = LaplacianMatrix(n_rx, n_rz, dist_r, dist_r)
    
    for i in range(nt_sub):
          a = lhs[:,:,i]#Uxxzz_GRAD_LAPLACE[:,:,i] #Uxx_GRAD_CST[:,:,i] + Uzz_GRAD_CST[:,:,i]
          A = a.flatten()
          F = np.eye(len(A),len(A)) * A             #scipy.sparse.dia_matrix((a, offset), shape=(rxx,rzz)).toarray()
          b = rhs[:,:,i].flatten()
          FF = np.matrix.getH(F)            #np.asmatrix(F).getH() #np.conj(F)
         
          LHS[:,:,i] = np.dot(FF,F)
          RHS[:,:,i] = np.dot(FF,b)
          
    # Damping
    a_s               = (np.eye(len(F),len(F))* 1e-30)      
    sum_LHS           = LHS.sum(2)    #+ a_s
    sum_RHS           = RHS.sum(2)    + (1e5 * a_s) # + (1e20 * a_s)
    
    inv_LHS           = np.linalg.inv(sum_LHS)
    inv_RHS           = np.linalg.inv(sum_RHS)
    
    g                 = (eps**2)*np.dot(Dxx_c,Dxx)
    inv_LHS_eps       = np.linalg.inv(sum_LHS + g)
    
    sum_inv_RHS       = np.diag(inv_RHS).sum(0)#(inv_RHS).sum(0).sum(0)
    sum_sum_LHS       = np.diag(sum_LHS).sum(0)#(sum_LHS).sum(0).sum(0)
    
    m0                = sum_inv_RHS * sum_sum_LHS  #### CARDINALITY???????? 1/((n_rx)**2) * sum_inv_RHS * sum_sum_LHS
    dm                = np.dot(inv_LHS_eps,(sum_RHS -(sum_LHS*(m0)) ))
    dm                = np.reshape(np.diag(dm), a.shape)
    
    c_GRAD_CST_inv  = np.sqrt(np.abs(dm+m0)) 

 
    
    return c_GRAD_CST_inv


def FCT_INV_COMPLEX(eps,n_rx, n_rz, lhs, rhs, nt_sub, dist_r):
           
    ##################################################################################################################################################
    #                               CONSTANT DENSITY MODEL 
    ##################################################################################################################################################
    
    # Initialization of variables 
    LHS      = np.zeros((n_rx**2, n_rz**2,nt_sub),dtype = 'complex_')
    RHS      = np.zeros((n_rx**2, n_rz**2,nt_sub),dtype = 'complex_')
    
    # Coefficient matrix
    # ex       = np.ones(n_rx)
    # data     = np.array([ex, -2 * ex, ex])
    # offsets  = np.array([-1, 0, 1])
    # Dxx      = scipy.sparse.dia_matrix((data, offsets), shape=(n_rx**2, n_rz**2)).toarray() * 1/dist_r**2
    # Dxx_c    = np.conj(Dxx)
    Dxx                 = LaplacianMatrix_CST(n_rx, n_rz, dist_r, dist_r)#.todense()
    Dxx_c    = np.matrix.getH(Dxx)
    
    
    
    # Objective function?
    #D = LaplacianMatrix(n_rx, n_rz, dist_r, dist_r)
    
    for i in range(nt_sub):
          a = lhs[:,:,i]#Uxxzz_GRAD_LAPLACE[:,:,i] #Uxx_GRAD_CST[:,:,i] + Uzz_GRAD_CST[:,:,i]
          A = a.flatten()
          F = np.eye(len(A),len(A)) * A             #scipy.sparse.dia_matrix((a, offset), shape=(rxx,rzz)).toarray()
          b = rhs[:,:,i].flatten()
          FF = np.matrix.getH(F)            #np.asmatrix(F).getH() #np.conj(F)
         
          LHS[:,:,i] = np.dot(FF,F)
          RHS[:,:,i] = np.dot(FF,b)
          
    # Damping
    a_s               = (np.eye(len(F),len(F))* 1e-30)      
    sum_LHS           = LHS.sum(2)    #+ a_s
    sum_RHS           = RHS.sum(2)     + (1e7 * a_s) #+ (1e20 * a_s)
    
    inv_LHS           = np.linalg.inv(sum_LHS)
    inv_RHS           = np.linalg.inv(sum_RHS)
    
    g                 = (eps**2)*np.dot(Dxx_c,Dxx)
    inv_LHS_eps       = np.linalg.inv(sum_LHS + g)
    
    sum_inv_RHS       = np.diag(inv_RHS).sum(0)#(inv_RHS).sum(0).sum(0)
    sum_sum_LHS       = np.diag(sum_LHS).sum(0)#(sum_LHS).sum(0).sum(0)
    
    m0                = sum_inv_RHS * sum_sum_LHS  #### CARDINALITY???????? 1/((n_rx)**2) * sum_inv_RHS * sum_sum_LHS
    dm                = np.dot(inv_LHS_eps,(sum_RHS -(sum_LHS*(m0)) ))
    dm                = np.reshape(np.diag(dm), a.shape)
    
    c_GRAD_CST_inv  = np.sqrt(dm+m0)

 
    
    return c_GRAD_CST_inv


def FCT_INV(eps,n_rx, n_rz, Uxxzz_GRAD_LAPLACE, Uxxzz_GRAD_LAPLACE_VAR, Uxx_GRAD_CST, Uzz_GRAD_CST, Utt_GRAD_CST, Uxx_GRAD_VAR, Uzz_GRAD_VAR, Utt_GRAD_VAR, c_m_sub, nt_sub, dist_r, mask):
           
    ##################################################################################################################################################
    #                               CONSTANT DENSITY MODEL 
    ##################################################################################################################################################
    
    # Initialization of variables 
    LHS      = np.zeros((n_rx**2, n_rz**2,nt_sub))
    RHS      = np.zeros((n_rx**2, n_rz**2,nt_sub))
    
    # Coefficient matrix
    # ex       = np.ones(n_rx)
    # data     = np.array([ex, -2 * ex, ex])
    # offsets  = np.array([-1, 0, 1])
    # Dxx      = scipy.sparse.dia_matrix((data, offsets), shape=(n_rx**2, n_rz**2)).toarray() * 1/dist_r**2
    # Dxx_c    = np.conj(Dxx)
    Dxx                 = LaplacianMatrix_CST(n_rx, n_rz, dist_r, dist_r)#.todense()
    Dxx_c    = np.matrix.getH(Dxx)
    
    
    
    # Objective function?
    #D = LaplacianMatrix(n_rx, n_rz, dist_r, dist_r)
    
    for i in range(nt_sub):
          a = Uxx_GRAD_CST[:,:,i] + Uzz_GRAD_CST[:,:,i]#Uxxzz_GRAD_LAPLACE[:,:,i] #Uxx_GRAD_CST[:,:,i] + Uzz_GRAD_CST[:,:,i]
          A = a.flatten()
          F = np.eye(len(A),len(A)) * A             #scipy.sparse.dia_matrix((a, offset), shape=(rxx,rzz)).toarray()
          b = Utt_GRAD_CST[:,:,i].flatten()
          FF = np.matrix.getH(F)            #np.asmatrix(F).getH() #np.conj(F)
         
          LHS[:,:,i] = np.dot(FF,F)
          RHS[:,:,i] = np.dot(FF,b)
          
    # Damping
    a_s               = (np.eye(len(F),len(F))* 1e-30)      
    sum_LHS           = LHS.sum(2)    #+ a_s
    sum_RHS           = RHS.sum(2)    + (1e20 * a_s)
    
    inv_LHS           = np.linalg.inv(sum_LHS)
    inv_RHS           = np.linalg.inv(sum_RHS)
    
    g                 = (eps**2)*np.dot(Dxx_c,Dxx)
    inv_LHS_eps       = np.linalg.inv(sum_LHS + g)
    
    sum_inv_RHS       = np.diag(inv_RHS).sum(0)#(inv_RHS).sum(0).sum(0)
    sum_sum_LHS       = np.diag(sum_LHS).sum(0)#(sum_LHS).sum(0).sum(0)
    
    m0                = sum_inv_RHS * sum_sum_LHS  #### CARDINALITY???????? 1/((n_rx)**2) * sum_inv_RHS * sum_sum_LHS
    dm                = np.dot(inv_LHS_eps,(sum_RHS -(sum_LHS*(m0)) ))
    dm                = np.reshape(np.diag(dm), a.shape)
    
    c_GRAD_CST_inv  = np.sqrt(np.abs(dm+m0)) 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Velocity Model - inversion output')
    plt.imshow(c_GRAD_CST_inv[1:-1,1:-1],vmin=np.min(c_GRAD_CST_inv[1:-1,1:-1]), vmax=np.max(c_GRAD_CST_inv[1:-1,1:-1]), aspect='auto', interpolation='none' ) #,extent=[-3, 3, -3, 3]
    plt.colorbar()     
    plt.show()    
    
    
    if mask == 'true':
        
        diff_sq = (c_m_sub[1:-1,1:-1] - c_GRAD_CST_inv[1:-1,1:-1])**2
        diff_sq = diff_sq.flatten()
        mask_   = diff_sq < 10000
        diff_sq = diff_sq * mask_
        diff_norm = np.sqrt(np.reshape(diff_sq, c_GRAD_CST_inv[1:-1,1:-1].shape))
        
        RMS     = np.sqrt((1/(n_rx-2)**2) * (diff_sq).sum(0))
        
        fig = plt.figure(figsize=(6,5))
        
        plt.imshow(diff_norm, vmin=0, vmax=np.max(diff_norm), aspect='auto', interpolation='none' )
        plt.colorbar()  
        plt.title('Difference true and recovered velocity, RMS=%f' %np.round(RMS ,6))   
        plt.xlabel('X Receiver number')
        plt.ylabel('Z Receiver number')   
        plt.show() 
    
    else: 
        
        diff_sq = (c_m_sub[1:-1,1:-1] - c_GRAD_CST_inv[1:-1,1:-1])**2
        diff_sq = diff_sq.flatten()
        diff_norm = np.sqrt(np.reshape(diff_sq, c_GRAD_CST_inv[1:-1,1:-1].shape)) 
        
        RMS = np.sqrt((1/(n_rx-2)**2) * ((c_m_sub[1:-1,1:-1] - c_GRAD_CST_inv[1:-1,1:-1])**2).sum(0).sum(0))
        
        plt.imshow(diff_norm, vmin=0, vmax=np.max(diff_norm), aspect='auto', interpolation='none' )
        plt.colorbar()  
        plt.title('Difference true and recovered velocity, RMS=%f' %np.round(RMS ,6))   
        plt.xlabel('X Receiver number')
        plt.ylabel('Z Receiver number')   
        plt.show() 
    
    
    ##################################################################################################################################################
    #                               VARIABLE DENSITY MODEL 
    ##################################################################################################################################################
    
    # Initialization of variables 
    LHS      = np.zeros((n_rx**2, n_rz**2,nt_sub))
    RHS      = np.zeros((n_rx**2, n_rz**2,nt_sub))
    
    # Coefficient matrix
    # ex       = np.ones(n_rx)
    # data     = np.array([ex, -2 * ex, ex])
    # offsets  = np.array([-1, 0, 1])
    # Dxx      = scipy.sparse.dia_matrix((data, offsets), shape=(n_rx**2, n_rz**2)).toarray() * 1/dist_r**2
    # Dxx_c    = np.conj(Dxx)
    
    #eps = 5
    
    # Objective function?
    #D = LaplacianMatrix(n_rx, n_rz, dist_r, dist_r)
    
    for i in range(nt_sub):
          a =  Uxxzz_GRAD_LAPLACE_VAR[:,:,i] #Uxx_GRAD_VAR[:,:,i] + Uzz_GRAD_VAR[:,:,i]
          A = a.flatten()
          F = np.eye(len(A),len(A)) * A             #scipy.sparse.dia_matrix((a, offset), shape=(rxx,rzz)).toarray()
          b = Utt_GRAD_VAR[:,:,i].flatten()
          FF = np.matrix.getH(F)            #np.asmatrix(F).getH() #np.conj(F)
         
          LHS[:,:,i] = np.dot(FF,F)
          RHS[:,:,i] = np.dot(FF,b)
          
    # Damping
    a_s               = (np.eye(len(F),len(F))* 1e-30)      
    sum_LHS           = LHS.sum(2)    + a_s
    sum_RHS           = RHS.sum(2)    + (1e24 * a_s)
    
    inv_LHS           = np.linalg.inv(sum_LHS)
    inv_RHS           = np.linalg.inv(sum_RHS)
    
    #g                 = (eps**2)*np.dot(Dxx_c,Dxx)
    #inv_LHS           = np.linalg.inv(sum_LHS + g)
    
    sum_inv_RHS       = np.diag(inv_RHS).sum(0)#(inv_RHS).sum(0).sum(0)
    sum_sum_LHS       = np.diag(sum_LHS).sum(0)#(sum_LHS).sum(0).sum(0)
    
    m0                =  sum_inv_RHS * sum_sum_LHS  #### CARDINALITY????????
    dm                = inv_LHS*(sum_RHS -(sum_LHS*(m0)) )
    dm                = np.reshape(np.diag(dm), a.shape)
    
    c_GRAD_VAR_inv  = np.sqrt(np.abs(dm+m0)) 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Velocity Model - inversion output')
    plt.imshow(c_GRAD_VAR_inv[1:-1,1:-1],vmin=np.min(c_GRAD_VAR_inv[1:-1,1:-1]), vmax=np.max(c_GRAD_VAR_inv[1:-1,1:-1]), aspect='auto', interpolation='none' ) #,extent=[-3, 3, -3, 3]
    plt.colorbar()     
    plt.show()    
    
    if mask == 'true':
        
        diff_sq = (c_m_sub[1:-1,1:-1] - c_GRAD_VAR_inv[1:-1,1:-1])**2
        diff_sq = diff_sq.flatten()
        mask_   = diff_sq < 10000
        diff_sq = diff_sq * mask_
        diff_norm = np.sqrt(np.reshape(diff_sq, c_GRAD_VAR_inv[1:-1,1:-1].shape))
        
        RMS     = np.sqrt((1/(n_rx-2)**2) * (diff_sq).sum(0))
        
        fig = plt.figure(figsize=(6,5))
        
        plt.imshow(diff_norm, vmin=0, vmax=np.max(diff_norm), aspect='auto', interpolation='none' )
        plt.colorbar()  
        plt.title('Difference true and recovered velocity, RMS=%f' %np.round(RMS ,6))   
        plt.xlabel('X Receiver number')
        plt.ylabel('Z Receiver number')   
        plt.show() 
    
    else: 
        
        diff_sq = (c_m_sub[1:-1,1:-1] - c_GRAD_VAR_inv[1:-1,1:-1])**2
        diff_sq = diff_sq.flatten()
        diff_norm = np.sqrt(np.reshape(diff_sq, c_GRAD_VAR_inv[1:-1,1:-1].shape)) 
        
        RMS = np.sqrt((1/(n_rx-2)**2) * ((c_m_sub[1:-1,1:-1] - c_GRAD_VAR_inv[1:-1,1:-1])**2).sum(0).sum(0))
        
        plt.imshow(diff_norm, vmin=0, vmax=np.max(diff_norm), aspect='auto', interpolation='none' )
        plt.colorbar()  
        plt.title('Difference true and recovered velocity, RMS=%f' %np.round(RMS ,6))   
        plt.xlabel('X Receiver number')
        plt.ylabel('Z Receiver number')   
        plt.show() 
    
    
    ##################################################################################################################################################
    #                              DIFFERENCE CST & VAR DENSITY MODEL 
    ##################################################################################################################################################
    
    rho_sig_inv = c_GRAD_VAR_inv[1:-1,1:-1] - c_GRAD_CST_inv[1:-1,1:-1]
    mean_RS = np.mean(np.abs(rho_sig_inv))
    
    fig = plt.figure(figsize=(6,5))
        
    plt.imshow(rho_sig_inv, vmin=np.min(rho_sig_inv), vmax=np.max(rho_sig_inv), aspect='auto', interpolation='none' )
    plt.colorbar()  
    plt.title('Density signal, mean difference={np.round(mean_RS ,6)} [m/s]')   
    plt.xlabel('X Receiver number')
    plt.ylabel('Z Receiver number')   
    plt.show()     
    
    return c_GRAD_CST_inv, c_GRAD_VAR_inv, rho_sig_inv
