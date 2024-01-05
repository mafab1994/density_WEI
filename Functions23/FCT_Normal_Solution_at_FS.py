# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np


def NORM_SOL_3D_ELASTIC_at_FS(dttv1_internal, dttv2_internal, dttv3_internal, A1_internal, A2_internal, A3_internal,\
                              B1_internal, B2_internal, B3_internal, nt_sub, nrx, nry, nrz, comp):
    
    N_internal        = dttv3_internal[:,:,0].flatten().shape[0]
    dttv3_internal    = np.reshape(dttv3_internal, (N_internal,nt_sub))
    dttv2_internal    = np.reshape(dttv2_internal, (N_internal,nt_sub))
    dttv1_internal    = np.reshape(dttv1_internal, (N_internal,nt_sub))
    A3_internal = np.reshape(A3_internal, (N_internal,nt_sub))
    B3_internal  = np.reshape(B3_internal, (N_internal,nt_sub))
    A2_internal = np.reshape(A2_internal, (N_internal,nt_sub))
    B2_internal  = np.reshape(B2_internal, (N_internal,nt_sub))
    A1_internal = np.reshape(A1_internal, (N_internal,nt_sub))
    B1_internal  = np.reshape(B1_internal, (N_internal,nt_sub))
    
    if comp == 'X':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)),A1_internal[i,:],-B1_internal[i,:]]).T
            y_rhs = dttv1_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-2,nry-2))
        vp_est = np.reshape(body_est[:,1], (nrx-2,nry-2))
        
    elif comp == 'Y':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)),A2_internal[i,:],-B2_internal[i,:]]).T
            y_rhs = dttv2_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-2,nry-2))
        vp_est = np.reshape(body_est[:,1], (nrx-2,nry-2))    
    
    elif comp == 'Z':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)),A3_internal[i,:],-B3_internal[i,:]]).T
            y_rhs = dttv3_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-2,nry-2))
        vp_est = np.reshape(body_est[:,1], (nrx-2,nry-2))    
    
    return vp_est, vs_est,X,y_rhs,inv,XX,SOL,body_est


def NORM_SOL_3D_ELASTIC_notCORR(dttv1_internal, dttv2_internal, dttv3_internal, rotROT1_internal, rotROT2_internal, rotROT3_internal,\
                        gradDIV1_internal, gradDIV2_internal, gradDIV3_internal, nt_sub, nrx, nry, nrz, comp,order):
    
    N_internal        = dttv3_internal[:,:,0].flatten().shape[0]
    dttv3_internal    = np.reshape(dttv3_internal, (N_internal,nt_sub))
    dttv2_internal    = np.reshape(dttv2_internal, (N_internal,nt_sub))
    dttv1_internal    = np.reshape(dttv1_internal, (N_internal,nt_sub))
    gradDIV3_internal = np.reshape(gradDIV3_internal, (N_internal,nt_sub))
    rotROT3_internal  = np.reshape(rotROT3_internal, (N_internal,nt_sub))
    gradDIV2_internal = np.reshape(gradDIV2_internal, (N_internal,nt_sub))
    rotROT2_internal  = np.reshape(rotROT2_internal, (N_internal,nt_sub))
    gradDIV1_internal = np.reshape(gradDIV1_internal, (N_internal,nt_sub))
    rotROT1_internal  = np.reshape(rotROT1_internal, (N_internal,nt_sub))
    
    if comp == 'X':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)), gradDIV1_internal[i,:],-rotROT1_internal[i,:]]).T
            y_rhs = dttv1_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-order,nry-order))
        vp_est = np.reshape(body_est[:,1], (nrx-order,nry-order))
        
    elif comp == 'Y':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)),gradDIV2_internal[i,:],-rotROT2_internal[i,:]]).T
            y_rhs = dttv2_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-order,nry-order))
        vp_est = np.reshape(body_est[:,1], (nrx-order,nry-order))    
    
    elif comp == 'Z':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)),gradDIV3_internal[i,:],-rotROT3_internal[i,:]]).T
            y_rhs = dttv3_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-order,nry-order))
        vp_est = np.reshape(body_est[:,1], (nrx-order,nry-order))    
    
    return vp_est, vs_est
