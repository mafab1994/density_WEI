# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_rho_inv_corr(U_filter,nr,n_rx, n_rz, N, nt_sub,ww_corner, ww_BC, ww_internal):
    
    M_l      = np.zeros((N,N, nt_sub))
    WW       = np.zeros((N,N))# *ww_BC 
    vL = np.arange(n_rx, N-n_rx, n_rx)
    vR = np.arange((2*n_rx)-1, N-n_rx, n_rx)
    
    for n in range(0,nt_sub):
        U = U_filter[:,:,n].flatten()
        j = 0
        for i in range(0,1):
            M_l[i,j,n] = -2* U[i] + U[i+1] + U[i+n_rx]#-4*
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr, n] =  U[i+nr] - U[i]
            
            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            WW[i,j+nr] =  ww_corner

        for i in range(1,nr-1):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -3* U[i] + U[i-1]+ U[i+1] + U[i+nr]#-4*
            M_l[i,j+2, n] =  U[i+1] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr+1] - U[i]
    
            WW[i,j] = ww_BC
            WW[i,j+1] = ww_BC
            WW[i,j+2] =  ww_BC
            WW[i,j+nr+1] =  ww_BC

            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -2* U[i] + U[i-1] + U[i+nr] #-4*
            M_l[i,j+nr+1, n] =  U[i+nr] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            WW[i,j+nr+1] =  ww_corner

            
        j = 0

        for i in range(nr, N-nr):
            if np.any(i == vL[:])==True:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr,n] = -3* U[i] + U[i-nr]+ U[i+1] + U[i+nr]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr+1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -3* U[i] + U[i-nr]+ U[i-1] + U[i+nr]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr-1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
                
                j=j+1

            else:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i+nr]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]

                WW[i,j] = ww_internal
                WW[i,j+nr-1] = ww_internal
                WW[i,j+nr] =  ww_internal
                WW[i,j+nr+1] =  ww_internal
                WW[i,j+(2*nr)] =  ww_internal
                j=j+1


        for i in range(N-nr, N-nr+1):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr,n] = -2* U[i] + U[i-nr] + U[i+1] # last term correct???
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+nr+1] =  ww_corner

            j=j+1


        for i in range(N-n_rx+1, N-1):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -3* U[i] + U[i-nr]+ U[i-1] + U[i+1] 
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]
            WW[i,j] = ww_BC
            WW[i,j+nr] = ww_BC
            WW[i,j+nr+1] =  ww_BC
            WW[i,j+nr-1] =  ww_BC
            j=j+1


        for i in range(N-1, N):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -2* U[i] + U[i-nr]+ U[i-1] 

            WW[i,j] = ww_corner
            WW[i,j+nr-1] = ww_corner
            WW[i,j+nr] =  ww_corner
    #M_l = M_l.sum(2)

    return M_l, WW
