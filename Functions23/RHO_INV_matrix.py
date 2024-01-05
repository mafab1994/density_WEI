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
            #j=j+1 #NOOO change 11.04.22

        for i in range(1,nr-1):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -3* U[i] + U[i-1]+ U[i+1] + U[i+nr]#-4*
            M_l[i,j+2, n] =  U[i+1] - U[i]
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr+1] - U[i]
    
            WW[i,j] = ww_BC
            WW[i,j+1] = ww_BC
            WW[i,j+2] =  ww_BC
            WW[i,j+nr+1] =  ww_BC

            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -2* U[i] + U[i-1] + U[i+nr] #-4*
            #M_l[i,j+nr+1, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            #WW[i,j+nr+1] =  ww_corner
            WW[i,j+nr+1] =  ww_corner
#             j=j+1
            
#         for i in range(nr, nr+1):
#             M_l[i,j-nr,n] =  U[i-nr] - U[i]
#             M_l[i,j+1,n] = -4* U[i] + U[i-1] + U[i+nr]
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]


            
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

# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_relative_RHO(U_filter,nr,n_rx, n_rz, N, nt_sub,ww_corner, ww_BC, ww_internal):
    
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
            #j=j+1 #NOOO change 11.04.22

        for i in range(1,nr-1):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -3* U[i] + U[i-1]+ U[i+1] + U[i+nr]#-4*
            M_l[i,j+2, n] =  U[i+1] - U[i]
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr+1] - U[i]
    
            WW[i,j] = ww_BC
            WW[i,j+1] = ww_BC
            WW[i,j+2] =  ww_BC
            WW[i,j+nr+1] =  ww_BC

            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -2* U[i] + U[i-1] + U[i+nr] #-4*
            #M_l[i,j+nr+1, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            #WW[i,j+nr+1] =  ww_corner
            WW[i,j+nr+1] =  ww_corner
#             j=j+1
            
#         for i in range(nr, nr+1):
#             M_l[i,j-nr,n] =  U[i-nr] - U[i]
#             M_l[i,j+1,n] = -4* U[i] + U[i-1] + U[i+nr]
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]


            
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
                M_l[i,j,n] = U[i-nr] + (2*U[i]) - U[i+nr]        #j-1
                M_l[i,j+nr-1,n] = U[i-1] + (2* U[i]) - U[i+1]     #i-1
                M_l[i,j+nr, n] =  (8* U[i]) + (2*U[i-nr]) + (2*U[i-1]) - (2*U[i+1]) - (2*U[i+nr]) #ij
                M_l[i,j+nr+1, n] =  - U[i-1] - (2* U[i]) + U[i+1]     #i+1
                M_l[i,j+(2*nr), n] =  -U[i-nr] - (2*U[i]) + U[i+nr]#j+1

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

# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_rho_inv_corrRHO(U_filter,rhom,nr,n_rx, n_rz, N, nt_sub,ww_corner, ww_BC, ww_internal):
    
    M_l      = np.zeros((N,N, nt_sub))
    WW       = np.zeros((N,N))# *ww_BC 
    vL = np.arange(n_rx, N-n_rx, n_rx)
    vR = np.arange((2*n_rx)-1, N-n_rx, n_rx)
    
    rho=rhom[:,:].flatten()
    for n in range(0,nt_sub):
        U = U_filter[:,:,n].flatten()
        j = 0
        for i in range(0,1):
            M_l[i,j,n] = (-2* rho[i]* U[i]) + (rho[i+1]* U[i+1]) + (rho[i+n_rx]*U[i+n_rx])#-4*
            M_l[i,j+1, n] =  (rho[i+1]*U[i+1]) - (rho[i]*U[i])
            M_l[i,j+nr, n] =  (rho[i+nr] * U[i+nr]) - (rho[i]*U[i])
            
            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            WW[i,j+nr] =  ww_corner
            #j=j+1 #NOOO change 11.04.22

        for i in range(1,nr-1):
            M_l[i,j,n] = (rho[i-1]*U[i-1]) - (rho[i]*U[i])
            M_l[i,j+1,n] = (-3* rho[i]*U[i]) + (rho[i-1]*U[i-1])+ (rho[i+1]*U[i+1]) + (rho[i+nr]*U[i+nr]) #-4*
            M_l[i,j+2, n] =  (rho[i+1]*U[i+1]) - (rho[i]*U[i])
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] =  (rho[i+nr+1]*U[i+nr+1]) - (rho[i]*U[i])
    
            WW[i,j] = ww_BC
            WW[i,j+1] = ww_BC
            WW[i,j+2] =  ww_BC
            WW[i,j+nr+1] =  ww_BC

            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j,n] = (rho[i-1]*U[i-1]) - (rho[i]*U[i])
            M_l[i,j+1,n] = (-2* rho[i]*U[i]) + (rho[i-1]*U[i-1]) + (rho[i+nr]*U[i+nr]) #-4*
            #M_l[i,j+nr+1, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] = (rho[i+nr]* U[i+nr]) - (rho[i]* U[i])

            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            #WW[i,j+nr+1] =  ww_corner
            WW[i,j+nr+1] =  ww_corner
#             j=j+1
            
#         for i in range(nr, nr+1):
#             M_l[i,j-nr,n] =  U[i-nr] - U[i]
#             M_l[i,j+1,n] = -4* U[i] + U[i-1] + U[i+nr]
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]


            
        j = 0

        for i in range(nr, N-nr):
            if np.any(i == vL[:])==True:
                M_l[i,j,n] = (rho[i-nr]*U[i-nr]) -(rho[i]*U[i])
                M_l[i,j+nr,n] = (-3* rho[i]*U[i]) + (rho[i-nr]*U[i-nr]) + (rho[i+1]*U[i+1]) + (rho[i+nr]* U[i+nr])
                M_l[i,j+nr+1, n] = (rho[i+1]* U[i+1]) - (rho[i]*U[i])
                M_l[i,j+(2*nr), n] =  (rho[i+nr]*U[i+nr]) - (rho[i]*U[i])
                
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr+1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j,n] = (rho[i-nr]*U[i-nr]) - (rho[i]*U[i])
                M_l[i,j+nr-1,n] = (rho[i-1]*U[i-1]) - (rho[i]*U[i])
                M_l[i,j+nr, n] =  (-3* rho[i]*U[i]) +(rho[i-nr] *U[i-nr])+ (rho[i-1]*U[i-1]) + (rho[i+nr]*U[i+nr])
                M_l[i,j+(2*nr), n] =  (rho[i+nr]*U[i+nr]) - (rho[i]*U[i])
                
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr-1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
                
                j=j+1

            else:
                M_l[i,j,n] = (rho[i-nr]*U[i-nr]) - (rho[i]*U[i])
                M_l[i,j+nr-1,n] = (rho[i-1]*U[i-1]) - (rho[i]*U[i])
                M_l[i,j+nr, n] =  (-4*rho[i]* U[i]) + (rho[i-nr]*U[i-nr]) + (rho[i-1]*U[i-1]) + (rho[i+1]*U[i+1]) + (rho[i+nr]*U[i+nr])
                M_l[i,j+nr+1, n] =  (rho[i+1]*U[i+1]) - (rho[i]*U[i])
                M_l[i,j+(2*nr), n] =  (rho[i+nr]*U[i+nr]) - (rho[i]*U[i])

                WW[i,j] = ww_internal
                WW[i,j+nr-1] = ww_internal
                WW[i,j+nr] =  ww_internal
                WW[i,j+nr+1] =  ww_internal
                WW[i,j+(2*nr)] =  ww_internal
                j=j+1


        for i in range(N-nr, N-nr+1):
            M_l[i,j,n] = (rho[i-nr]*U[i-nr]) - (rho[i]*U[i])
            M_l[i,j+nr,n] = (-2*rho[i]* U[i]) + (rho[i-nr]*U[i-nr]) + (rho[i+1]*U[i+1]) # last term correct???
            M_l[i,j+nr+1, n] =  (rho[i+1]*U[i+1]) - (rho[i]*U[i])

            WW[i,j] = ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+nr+1] =  ww_corner

            j=j+1


        for i in range(N-n_rx+1, N-1):
            M_l[i,j,n] = (rho[i-nr]*U[i-nr]) - (rho[i]*U[i])
            M_l[i,j+nr-1,n] = (rho[i-1]*U[i-1]) - (rho[i]*U[i])
            M_l[i,j+nr, n] =  (-3*rho[i]* U[i]) + (rho[i-nr]*U[i-nr])+ (rho[i-1]*U[i-1]) + (rho[i+1]*U[i+1]) 
            M_l[i,j+nr+1, n] =  (rho[i+1]*U[i+1]) - (rho[i]*U[i])
            WW[i,j] = ww_BC
            WW[i,j+nr] = ww_BC
            WW[i,j+nr+1] =  ww_BC
            WW[i,j+nr-1] =  ww_BC
            j=j+1


        for i in range(N-1, N):
            M_l[i,j,n] = (rho[i-nr]*U[i-nr]) - (rho[i]*U[i])
            M_l[i,j+nr-1,n] = (rho[i-1]*U[i-1]) - (rho[i]*U[i])
            M_l[i,j+nr, n] =  (-2*rho[i]* U[i]) + (rho[i-nr]*U[i-nr])+ (rho[i-1]*U[i-1]) 

            WW[i,j] = ww_corner
            WW[i,j+nr-1] = ww_corner
            WW[i,j+nr] =  ww_corner
    #M_l = M_l.sum(2)

    return M_l, WW

# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_rho_inv(U_filter,nr,n_rx, n_rz, N, nt_sub,ww_corner, ww_BC, ww_internal):
    
    M_l      = np.zeros((N,N, nt_sub))
    WW       = np.ones((N,N)) *ww_BC 
    vR = np.arange(n_rx, N-n_rx, n_rx)
    vL = np.arange((2*n_rx)-1, N-n_rx, n_rx)
    
    for n in range(0,nt_sub):
        U = U_filter[:,:,n].flatten()
        j = 0
        for i in range(0,1):
            M_l[i,j,n] = -4* U[i] + U[i+1] + U[i+n_rx]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr, n] =  U[i+nr] - U[i]
            
            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            WW[i,j+nr] =  ww_corner
            j=j+1 

        for i in range(1,nr-1):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -4* U[i] + U[i-1]+ U[i+1] + U[i+nr]
            M_l[i,j+2, n] =  U[i+1] - U[i]
            M_l[i,j+nr, n] =  U[i+nr] - U[i]
            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -4* U[i] + U[i-1] + U[i+nr]
            M_l[i,j+nr+1, n] =  U[i+nr] - U[i]
            M_l[i,j+nr, n] =  U[i+nr] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            WW[i,j+nr+1] =  ww_corner
            WW[i,j+nr] =  ww_corner
            
        j = 0

        for i in range(nr, N-nr):
            if np.any(i == vL[:])==True:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr,n] = -4* U[i] + U[i-nr]+ U[i+1] + U[i+nr]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+nr]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
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
            M_l[i,j+nr,n] = -4* U[i] + U[i-nr] + U[i]
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+nr+1] =  ww_corner

            j=j+1


        for i in range(N-n_rx+1, N-1):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+1] 
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]

            j=j+1


        for i in range(N-1, N):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] 

            WW[i,j] = ww_corner
            WW[i,j+nr-1] = ww_corner
            WW[i,j+nr] =  ww_corner
    #M_l = M_l.sum(2)

    return M_l, WW


# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_rho_inv_corr_ELASTIC(U_filter,nr,n_rx, n_rz, N, nt_sub,ww_corner, ww_BC, ww_internal):
    
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
            #j=j+1 #NOOO change 11.04.22

        for i in range(1,nr-1):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -3* U[i] + U[i-1]+ U[i+1] + U[i+nr]#-4*
            M_l[i,j+2, n] =  U[i+1] - U[i]
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr+1] - U[i]
    
            WW[i,j] = ww_BC
            WW[i,j+1] = ww_BC
            WW[i,j+2] =  ww_BC
            WW[i,j+nr+1] =  ww_BC

            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -2* U[i] + U[i-1] + U[i+nr] #-4*
            #M_l[i,j+nr+1, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            #WW[i,j+nr+1] =  ww_corner
            WW[i,j+nr+1] =  ww_corner
#             j=j+1
            
#         for i in range(nr, nr+1):
#             M_l[i,j-nr,n] =  U[i-nr] - U[i]
#             M_l[i,j+1,n] = -4* U[i] + U[i-1] + U[i+nr]
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]


            
        j = 0

        for i in range(nr, N-nr):
            if np.any(i == vL[:])==True:
                M_l[i,j,n] = U[i+nr] - U[i]
                M_l[i,j+nr,n] = -3* U[i] + U[i+nr]+ U[i-1] + U[i-nr]
                M_l[i,j+nr+1, n] =  U[i-1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i-nr] - U[i]
                
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr+1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -3* U[i] + U[i+nr]+ U[i+1] + U[i-nr]
                M_l[i,j+(2*nr), n] =  U[i-nr] - U[i]
                
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr-1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
                
                j=j+1

            else:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -4* U[i] + U[i+nr]+ U[i+1] + U[i-1] + U[i-nr]
                M_l[i,j+nr+1, n] =  U[i-1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i-nr] - U[i]

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

# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_rho_inv2(U_filter,nr,n_rx, n_rz, N, nt_sub,ww_corner, ww_BC, ww_internal):
    
    M_l      = np.zeros((N,N, nt_sub))
    WW       = np.ones((N,N)) *ww_BC 
    vR = np.arange(n_rx, N-n_rx, n_rx)
    vL = np.arange((2*n_rx)-1, N-n_rx, n_rx)
    
    for n in range(0,nt_sub):
        U = U_filter[:,:,n].flatten()
        j = 0
        for i in range(0,1):
            M_l[i,j,n] = -4* U[i] + U[i+1] + U[i+n_rx]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr, n] =  U[i+nr] - U[i]
            
            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            WW[i,j+nr] =  ww_corner
            j=j+1

        for i in range(1,nr-1):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -4* U[i] + U[i-1]+ U[i+1] + U[i+nr]
            M_l[i,j+2, n] =  U[i+1] - U[i]
            M_l[i,j+nr, n] =  U[i+nr] - U[i]
            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -4* U[i] + U[i-1] + U[i+nr]
            M_l[i,j+nr+1, n] =  U[i+nr] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            WW[i,j+nr+1] =  ww_corner
            j=j+1
            
        j = 0

        for i in range(nr, N-nr):
            if np.any(i == vL[:])==True:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr,n] = -4* U[i] + U[i-nr]+ U[i+1] + U[i+nr]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+nr]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
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
            M_l[i,j+nr,n] = -4* U[i] + U[i-nr] + U[i]
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+nr+1] =  ww_corner

            j=j+1


        for i in range(N-n_rx+1, N-1):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+1] 
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]

            j=j+1


        for i in range(N-1, N):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] 

            WW[i,j] = ww_corner
            WW[i,j+nr-1] = ww_corner
            WW[i,j+nr] =  ww_corner
    #M_l = M_l.sum(2)

    return M_l, WW

# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_rho_inv_3D(U, rxx, nrz, nt_sub, central_pt):
    N = rxx**2 *(nrz-2)
    M_l      = np.zeros((N,N, nt_sub))
    
    for n in range(0,nt_sub):
        j = 0

        for i in range(0, nrz-2):
            
                M_l[i,j,n] = U[central_pt+i-1,n] - U[central_pt+i,n]
                M_l[i,j+nrz,n] = U[central_pt+i-nrz,n] - U[central_pt+i,n]
                M_l[i,j+((rxx)*nrz),n] = U[central_pt+i-((rxx)*nrz),n] - U[central_pt+i,n]
                M_l[i,j+1,n] = - (6* U[central_pt+i,n]) + U[central_pt+i-1,n] + U[central_pt+i-nrz,n] + U[central_pt+i-((rxx)*nrz),n] \
                                + U[central_pt+i+1,n] +   U[central_pt+i+nrz,n] + U[central_pt+i+((rxx)*nrz),n] 
                M_l[i,j+2,n] = U[central_pt+i+1,n] - U[central_pt+i,n]
                M_l[i,j+(2*nrz),n] = U[central_pt+i+nrz,n] - U[central_pt+i,n]
                M_l[i,j+(2*(rxx)*nrz),n] = U[central_pt+i+((rxx)*nrz),n] - U[central_pt+i,n]
    

                j=j+1

    return M_l,N


# +
import numpy as np

def matrix_rho_inv_2D_(U_filter, nr, ww_internal, NN, nt_sub):
    M_l      = np.zeros((NN,NN, nt_sub))
    WW       = np.ones((NN,NN)) *ww_internal
    j=0
    for n in range(0,nt_sub):
        U = U_filter[:,:,n].flatten()
        for i in range(0, NN):
                    M_l[i,j,n] = U[i-nr] - U[i]
                    M_l[i,j+nr-1,n] = U[i-1] - U[i]
                    M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i+nr]
                    M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                    M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                    j=j+1
                
    return M_l, WW


# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_rho_inv_corr3D_old(U_filter,nr,nrx,nry, N, nt_sub,ww_corner, ww_BC, ww_internal):
    
    M_l      = np.zeros((N,N, nt_sub))
    WW       = np.zeros((N,N))# *ww_BC 
    vL = np.arange(nrx, N-nrx, nrx)
    vR = np.arange((2*nrx)-1, N-nrx, nrx)
    
    for n in range(0,nt_sub):
        U = U_filter[:,:,:,n].flatten()
        j = 0
        for i in range(0,1):
            M_l[i,j,n] = -3* U[i] + U[i+1] + U[i+nry] + U[i+(nry*nrx)]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr, n] =  U[i+nry] - U[i]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            WW[i,j+nr] =  ww_corner
            WW[i,j+(nry*nrx)] =  ww_corner
            #j=j+1 #NOOO change 11.04.22

        for i in range(1,nr-1):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -4* U[i] + U[i-1]+ U[i+1] + U[i+nr]
            M_l[i,j+2, n] =  U[i+1] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr] - U[i]
            M_l[i,j+(nry*nrx)+1, n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j] = ww_BC
            WW[i,j+1] = ww_BC
            WW[i,j+2] =  ww_BC
            WW[i,j+nr+1] =  ww_BC
            WW[i,j+(nry*nrx)+1] =  ww_BC
            
            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j,n] = U[i-1] - U[i]
            M_l[i,j+1,n] = -4* U[i] + U[i-1] + U[i+nr]
            #M_l[i,j+nr+1, n] =  U[i+nr] - U[i]
            M_l[i,j+nr+1, n] =  U[i+nr] - U[i]
            M_l[i,j+(nry*nrx)+1, n] =  U[i+(nry*nrx)] - U[i]

            WW[i,j] = ww_corner
            WW[i,j+1] = ww_corner
            #WW[i,j+nr+1] =  ww_corner
            WW[i,j+nr+1] =  ww_corner
            WW[i,j+(nry*nrx)+1] = ww_corner
#             j=j+1
            
#         for i in range(nr, nr+1):
#             M_l[i,j-nr,n] =  U[i-nr] - U[i]
#             M_l[i,j+1,n] = -4* U[i] + U[i-1] + U[i+nr]
#             M_l[i,j+nr, n] =  U[i+nr] - U[i]


            
        j = 0

        for i in range(nr, (nrx*nry)-nr):
            if np.any(i == vL[:])==True:
                
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr,n] = -4* U[i] + U[i-nr]+ U[i+1] + U[i+nr] +  U[i+(nry*nrx)]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]
                
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr+1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
                WW[i,j+(nry*nrx)+nr] = ww_BC

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+nr] +  U[i+(nry*nrx)]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]
                
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr-1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
                WW[i,j+(nry*nrx)+nr] = ww_BC
                
                j=j+1

            else:
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -5* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i+nr] +  U[i+(nry*nrx)]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j] = ww_internal
                WW[i,j+nr-1] = ww_internal
                WW[i,j+nr] =  ww_internal
                WW[i,j+nr+1] =  ww_internal
                WW[i,j+(2*nr)] =  ww_internal
                WW[i,j+(nry*nrx)+nr] = ww_internal

                j=j+1
               
        for i in range((nrx*nry)-nr, (nrx*nry)-nr+1):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr,n] = -3* U[i] + U[i-nr] + U[i] + U[i+(nrx*nry)]
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]
            M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]

            
            WW[i,j-(nrx*nry)+nr] =  ww_corner
            WW[i,j] = ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+nr+1] =  ww_corner
            WW[i,j+(nry*nrx)+nr] = ww_corner

            j=j+1


        for i in range((nrx*nry)-nr+1, (nrx*nry)-1):
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i-(nrx*nry)]
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]
            M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j] = ww_BC
            WW[i,j+nr] = ww_BC
            WW[i,j+nr+1] =  ww_BC
            WW[i,j+nr-1] =  ww_BC
            WW[i,j+(nry*nrx)+nr]  =  ww_BC
            j=j+1


        for i in range((nrx*nry)-1, (nrx*nry)):
            
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -3* U[i] + U[i-nr]+ U[i-1] + U[i-(nrx*nry)]
            M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j] = ww_corner
            WW[i,j+nr-1] = ww_corner
            WW[i,j+nr] =  ww_corner
            WW[i,j+(nrx*nry)+nr] =  ww_corner
            
    
        for i in range((nrx*nry), (nrx*nry)+1):
            
            M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr,n] = -4* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)+nr] + U[i+(nry*nrx)]
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]
            M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j-(nrx*nry)+nr] =  ww_corner
            WW[i,j] = ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+nr+1] =  ww_corner
            WW[i,j+(nrx*nry)+nr] =  ww_corner

            j=j+1
            
        
                        
                
        for i in range((nrx*nry)+1, N-(nrx*nry)-nr):
            
            if i in range((2*nrx*nry)-1, (2*nrx*nry)):
            
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -3* U[i] + U[i-nr]+ U[i-1] + U[i-(nrx*nry)]
                M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j] = ww_corner
                WW[i,j+nr-1] = ww_corner
                WW[i,j+nr] =  ww_corner
                WW[i,j+(nrx*nry)+nr] =  ww_corner
                
            if i in range((2*nrx*nry), (2*nrx*nry)+1):
            
                M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr,n] = -4* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j-(nrx*nry)+nr] =  ww_corner
                WW[i,j] = ww_corner
                WW[i,j+nr] = ww_corner
                WW[i,j+nr+1] =  ww_corner
                WW[i,j+(nrx*nry)+nr] =  ww_corner

                j=j+1
                
            if np.any(i == vL[:])==True:
                M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr,n] = -5* U[i] + U[i-nr]+ U[i+1] + U[i+nr] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]
                
                WW[i,j-(nrx*nry)+nr] = ww_BC
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr+1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
                WW[i,j+(nrx*nry)+nr] = ww_BC

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+nr] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]
                
                WW[i,j-(nrx*nry)+nr] = ww_BC
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr-1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
                WW[i,j+(nrx*nry)+nr] = ww_BC
                
                j=j+1

            else:
                M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i+nr] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx)+nr, n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j-(nrx*nry)+nr] = ww_internal
                WW[i,j] = ww_internal
                WW[i,j+nr-1] = ww_internal
                WW[i,j+nr] =  ww_internal
                WW[i,j+nr+1] =  ww_internal
                WW[i,j+(2*nr)] =  ww_internal
                WW[i,j+(nrx*nry)+nr] = ww_internal
                j=j+1
                
                
        for i in range(N-(nrx*nry)-nr, N-nr):
            
                
            if np.any(i == vL[:])==True:
                M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr,n] = -4* U[i] + U[i-nr]+ U[i+1] + U[i+nr] + U[i-(nrx*nry)] 
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                
                
                WW[i,j-(nrx*nry)+nr] = ww_BC
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr+1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
                

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -3* U[i] + U[i-nr]+ U[i-1] + U[i+nr] + U[i-(nrx*nry)] 
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                
                
                WW[i,j-(nrx*nry)+nr] = ww_BC
                WW[i,j] = ww_BC
                WW[i,j+nr] = ww_BC
                WW[i,j+nr-1] =  ww_BC
                WW[i,j+(2*nr)] =  ww_BC
               
                
                j=j+1

            else:
                M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j,n] = U[i-nr] - U[i]
                M_l[i,j+nr-1,n] = U[i-1] - U[i]
                M_l[i,j+nr, n] =  -5* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i+nr] + U[i-(nrx*nry)] 
                M_l[i,j+nr+1, n] =  U[i+1] - U[i]
                M_l[i,j+(2*nr), n] =  U[i+nr] - U[i]
                

                WW[i,j-(nrx*nry)+nr] = ww_internal
                WW[i,j] = ww_internal
                WW[i,j+nr-1] = ww_internal
                WW[i,j+nr] =  ww_internal
                WW[i,j+nr+1] =  ww_internal
                WW[i,j+(2*nr)] =  ww_internal
                j=j+1
                
                                


        for i in range(N-nr, N-nr+1):
            M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr,n] = -3* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)]
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]
            
            WW[i,j-(nrx*nry)+nr] =  ww_corner
            WW[i,j] = ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+nr+1] =  ww_corner

            j=j+1


        for i in range(N-nrx+1, N-1):
            M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -4* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i-(nrx*nry)]
            M_l[i,j+nr+1, n] =  U[i+1] - U[i]
            
            WW[i,j-(nrx*nry)+nr] =  ww_BC
            WW[i,j] = ww_BC
            WW[i,j+nr] = ww_BC
            WW[i,j+nr+1] =  ww_BC
            WW[i,j+nr-1] =  ww_BC
            j=j+1


        for i in range(N-1, N):
            M_l[i,j-(nrx*nry)+nr,n] = U[i-(nrx*nry)] - U[i]
            M_l[i,j,n] = U[i-nr] - U[i]
            M_l[i,j+nr-1,n] = U[i-1] - U[i]
            M_l[i,j+nr, n] =  -3* U[i] + U[i-nr]+ U[i-1] + U[i-(nrx*nry)]
            
            WW[i,j-(nrx*nry)+nr] =  ww_corner
            WW[i,j] = ww_corner
            WW[i,j+nr-1] = ww_corner
            WW[i,j+nr] =  ww_corner
    #M_l = M_l.sum(2)

    return M_l, WW

# +
from scipy import *
import scipy
import numpy as np
############################################################################################# 
#   System for all unknowns - Construction of A matrix on LHS
#############################################################################################

def matrix_rho_inv_corr3D_new(U_filter,nr,nrx,nry, N, nt_sub,ww_corner, ww_BC, ww_internal):
    
    M_l      = np.zeros((N,N, nt_sub))
    WW       = np.zeros((N,N))# *ww_BC
    Nxy      = nr**2
    vL = np.arange(nr, N-nr, nr)
    vR = np.arange((2*nr)-1, N-nr, nr)
    
    for n in range(0,nt_sub):
        U = U_filter[:,:,:,n].flatten()
        j = 0
        
        for i in range(0,1):
            M_l[i,j,n] = -6* U[i] + U[i+1] + U[i+nry] + U[i+(nry*nrx)]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr, n] =  U[i+nry] - U[i]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j]           = ww_corner
            WW[i,j+1]         = ww_corner
            WW[i,j+nr]        =  ww_corner
            WW[i,j+(nry*nrx)] =  ww_corner
            j=j+1 

        for i in range(1,nr-1):
            M_l[i,j-1,n] = U[i-1] - U[i]
            M_l[i,j,n] = -6* U[i] + U[i-1]+ U[i+1] + U[i+nr]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr, n] =  U[i+nr] - U[i]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j-1]         = ww_BC
            WW[i,j]           = ww_BC
            WW[i,j+1]         =  ww_BC
            WW[i,j+nr]        =  ww_BC
            WW[i,j+(nry*nrx)] =  ww_BC
            
            j=j+1


        for i in range(nr-1, nr):
            M_l[i,j-1,n] = U[i-1] - U[i]
            M_l[i,j,n] = -6* U[i] + U[i-1] + U[i+nr]
            M_l[i,j+nr, n] =  U[i+nr] - U[i]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]

            WW[i,j-1]         = ww_corner
            WW[i,j]           = ww_corner
            WW[i,j+nr]        =  ww_corner
            WW[i,j+(nry*nrx)] = ww_corner
            j=j+1

            

        for i in range(nr, (nrx*nry)-nr):
            if np.any(i == vL[:])==True:
                
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j,n] = -6* U[i] + U[i-nr]+ U[i+1] + U[i+nr] +  U[i+(nry*nrx)]
                M_l[i,j+1, n] =  U[i+1] - U[i]
                M_l[i,j+(nr), n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
                
                WW[i,j-nr] = ww_BC
                WW[i,j]    = ww_BC
                WW[i,j+1]  =  ww_BC
                WW[i,j+nr] =  ww_BC
                WW[i,j+(nry*nrx)] = ww_BC

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+nr] +  U[i+(nry*nrx)]
                M_l[i,j+nr, n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
                
                WW[i,j-nr] = ww_BC
                WW[i,j-1] = ww_BC
                WW[i,j] =  ww_BC
                WW[i,j+nr] =  ww_BC
                WW[i,j+(nry*nrx)] = ww_BC
                
                j=j+1

            else:
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i+nr] +  U[i+(nry*nrx)]
                M_l[i,j+1, n] =  U[i+1] - U[i]
                M_l[i,j+nr, n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j-nr] = ww_internal
                WW[i,j-1] = ww_internal
                WW[i,j] =  ww_internal
                WW[i,j+1] =  ww_internal
                WW[i,j+nr] =  ww_internal
                WW[i,j+(nry*nrx)] = ww_internal

                j=j+1
               
        for i in range((nrx*nry)-nr, (nrx*nry)-nr+1):
            M_l[i,j-nr,n] = U[i-nr] - U[i]
            M_l[i,j,n] = -6* U[i] + U[i-nr] + U[i] + U[i+(nrx*nry)]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]

            WW[i,j-nr] = ww_corner
            WW[i,j] = ww_corner
            WW[i,j+1] =  ww_corner
            WW[i,j+(nry*nrx)] = ww_corner

            j=j+1


        for i in range((nrx*nry)-nr+1, (nrx*nry)-1):
            M_l[i,j-nr,n] = U[i-nr] - U[i]
            M_l[i,j-1,n] = U[i-1] - U[i]
            M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i-(nrx*nry)]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j-nr] = ww_BC
            WW[i,j-1] = ww_BC
            WW[i,j] =  ww_BC
            WW[i,j+1] =  ww_BC
            WW[i,j+(nry*nrx)]  =  ww_BC
            j=j+1


        for i in range((nrx*nry)-1, (nrx*nry)):
            
            M_l[i,j-nr,n] = U[i-nr] - U[i]
            M_l[i,j-1,n] = U[i-1] - U[i]
            M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i-(nrx*nry)]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j-nr] = ww_corner
            WW[i,j-1] = ww_corner
            WW[i,j] =  ww_corner
            WW[i,j+(nrx*nry)] =  ww_corner
            
        ##until here z level is 0
        
        ## here starts z level 1
        
        j = (nrx*nry)
        for i in range((nrx*nry), (nrx*nry)+1):
            
            M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
            M_l[i,j,n] = -6* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr,n] = U[i+nr] - U[i]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j-(nrx*nry)] =  ww_corner
            WW[i,j] = ww_corner
            WW[i,j+1] =  ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+(nrx*nry)] =  ww_corner

            j=j+1
            
        for i in range((nrx*nry)+1, (nrx*nry)+nr-1):
            
            M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
            M_l[i,j-1, n] =  U[i-1] - U[i]
            M_l[i,j,n] = -6* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr,n] = U[i+nr] - U[i]
            M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
            
            WW[i,j-(nrx*nry)] =  ww_corner
            WW[i,j-1] = ww_corner
            WW[i,j] = ww_corner
            WW[i,j+1] =  ww_corner
            WW[i,j+nr] = ww_corner
            WW[i,j+(nrx*nry)] =  ww_corner

            j=j+1
            
        for i in range((nrx*nry)+nr-1, (nrx*nry)+nr):
            
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-1, n] =  U[i-1] - U[i]
                M_l[i,j,n] = -6* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+nr,n] = U[i+nr] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
            
                WW[i,j-(nrx*nry)] =  ww_corner
                WW[i,j-1] = ww_corner
                WW[i,j] =  ww_corner
                WW[i,j+nr] = ww_corner
                WW[i,j+(nrx*nry)] =  ww_corner

                j=j+1
                      
                
        #for i in range((nrx*nry)+1, N-(nrx*nry)-nr):
        for i in range((nrx*nry)+nr, 2*(nrx*nry)-nr):
            
                
            if np.any(i == vL[:])==True:
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j,n] = -6* U[i] + U[i-nr]+ U[i+1] + U[i+nr] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+1, n] =  U[i+1] - U[i]
                M_l[i,j+nr, n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
                
                WW[i,j-(nrx*nry)] = ww_BC
                WW[i,j-nr] = ww_BC
                WW[i,j] = ww_BC
                WW[i,j+1] =  ww_BC
                WW[i,j+nr] =  ww_BC
                WW[i,j+(nrx*nry)] = ww_BC

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+nr] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+nr, n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]
                
                WW[i,j-(nrx*nry)] = ww_BC
                WW[i,j-nr] = ww_BC
                WW[i,j-1] = ww_BC
                WW[i,j] =  ww_BC
                WW[i,j+nr] =  ww_BC
                WW[i,j+(nrx*nry)] = ww_BC
                
                j=j+1

            else:
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i+nr] + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+1, n] =  U[i+1] - U[i]
                M_l[i,j+nr, n] =  U[i+nr] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j-(nrx*nry)] = ww_internal
                WW[i,j-nr] = ww_internal
                WW[i,j-1] = ww_internal
                WW[i,j] =  ww_internal
                WW[i,j+1] =  ww_internal
                WW[i,j+nr] =  ww_internal
                WW[i,j+(nrx*nry)] = ww_internal
                j=j+1
        
        for i in range( 2*(nrx*nry)-nr, 2*(nrx*nry)-nr+1):
            
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr] + U[i+1]  + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+1, n] =  U[i+1] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j-(nrx*nry)] = ww_corner
                WW[i,j-nr] = ww_corner
                WW[i,j] =  ww_corner
                WW[i,j+1] =  ww_corner
                WW[i,j+(nrx*nry)] = ww_corner
                j=j+1
                
                
        for i in range( 2*(nrx*nry)-nr+1, 2*(nrx*nry)-1):
            
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+1]  + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+1, n] =  U[i+1] - U[i]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j-(nrx*nry)] = ww_BC
                WW[i,j-nr] = ww_BC
                WW[i,j] =  ww_BC
                WW[i,j+1] =  ww_BC
                WW[i,j+(nrx*nry)] = ww_BC
                j=j+1

        for i in range( 2*(nrx*nry)-1, 2*(nrx*nry)):
            
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr] + U[i-1]  + U[i-(nrx*nry)] + U[i+(nry*nrx)]
                M_l[i,j+(nry*nrx), n] =  U[i+(nry*nrx)] - U[i]

                WW[i,j-(nrx*nry)] = ww_corner
                WW[i,j-1] =  ww_corner
                WW[i,j-nr] = ww_corner
                WW[i,j] =  ww_corner
                WW[i,j+(nrx*nry)] = ww_corner
                j=j+1
                                
        
        ##until here z level is 1
        
        ## here starts z level 2  
        j = 2*(nrx*nry)
                
        for i in range(2*(nrx*nry), 2*(nrx*nry)+1):
            
            M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
            M_l[i,j,n] = -6* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)] 
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr,n] = U[i+nr] - U[i]
            
            
            WW[i,j-(nrx*nry)] =  ww_corner
            WW[i,j] = ww_corner
            WW[i,j+1] =  ww_corner
            WW[i,j+nr] = ww_corner
            
            j=j+1
            
        for i in range(2*(nrx*nry)+1, 2*(nrx*nry)+nr-1):
            
            M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
            M_l[i,j-1, n] =  U[i-1] - U[i]
            M_l[i,j,n] = -6* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)] 
            M_l[i,j+1, n] =  U[i+1] - U[i]
            M_l[i,j+nr,n] = U[i+nr] - U[i]
            
            WW[i,j-(nrx*nry)] =  ww_corner
            WW[i,j-1] = ww_corner
            WW[i,j] = ww_corner
            WW[i,j+1] =  ww_corner
            WW[i,j+nr] = ww_corner

            j=j+1
            
        for i in range(2*(nrx*nry)+nr-1, 2*(nrx*nry)+nr):
            
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-1, n] =  U[i-1] - U[i]
                M_l[i,j,n] = -6* U[i] + U[i-nr] + U[i] + U[i-(nrx*nry)] 
                M_l[i,j+nr,n] = U[i+nr] - U[i]
            
                WW[i,j-(nrx*nry)] =  ww_corner
                WW[i,j-1] = ww_corner
                WW[i,j] =  ww_corner
                WW[i,j+nr] = ww_corner

                j=j+1
                      
                
        for i in range(2*(nrx*nry)+nr, N-nr):
            
                
            if np.any(i == vL[:])==True:
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j,n] = -6* U[i] + U[i-nr]+ U[i+1] + U[i+nr] + U[i-(nrx*nry)] 
                M_l[i,j+1, n] =  U[i+1] - U[i]
                M_l[i,j+nr, n] =  U[i+nr] - U[i]
                
                WW[i,j-(nrx*nry)] = ww_BC
                WW[i,j-nr] = ww_BC
                WW[i,j] = ww_BC
                WW[i,j+1] =  ww_BC
                WW[i,j+nr] =  ww_BC

                j=j+1

            elif np.any(i == vR[:])==True:
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+nr] + U[i-(nrx*nry)] 
                M_l[i,j+nr, n] =  U[i+nr] - U[i]
                
                WW[i,j-(nrx*nry)] = ww_BC
                WW[i,j-nr] = ww_BC
                WW[i,j-1] = ww_BC
                WW[i,j] =  ww_BC
                WW[i,j+nr] =  ww_BC
                
                j=j+1

            else:
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+1] + U[i+nr] + U[i-(nrx*nry)] 
                M_l[i,j+1, n] =  U[i+1] - U[i]
                M_l[i,j+nr, n] =  U[i+nr] - U[i]

                WW[i,j-(nrx*nry)] = ww_internal
                WW[i,j-nr] = ww_internal
                WW[i,j-1] = ww_internal
                WW[i,j] =  ww_internal
                WW[i,j+1] =  ww_internal
                WW[i,j+nr] =  ww_internal
                j=j+1
        
        for i in range( N-nr,N-nr+1):
            
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr] + U[i+1]  + U[i-(nrx*nry)] 
                M_l[i,j+1, n] =  U[i+1] - U[i]

                WW[i,j-(nrx*nry)] = ww_corner
                WW[i,j-nr] = ww_corner
                WW[i,j] =  ww_corner
                WW[i,j+1] =  ww_corner
                j=j+1
                
                
        for i in range( N-nr+1, N-1):
            
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr]+ U[i-1] + U[i+1]  + U[i-(nrx*nry)] 
                M_l[i,j+1, n] =  U[i+1] - U[i]

                WW[i,j-(nrx*nry)] = ww_BC
                WW[i,j-nr] = ww_BC
                WW[i,j] =  ww_BC
                WW[i,j+1] =  ww_BC
                j=j+1

        for i in range( N-1, N):
            
                M_l[i,j-(nrx*nry),n] = U[i-(nrx*nry)] - U[i]
                M_l[i,j-nr,n] = U[i-nr] - U[i]
                M_l[i,j-1,n] = U[i-1] - U[i]
                M_l[i,j, n] =  -6* U[i] + U[i-nr] + U[i-1]  + U[i-(nrx*nry)] 

                WW[i,j-(nrx*nry)] = ww_corner
                WW[i,j-1] =  ww_corner
                WW[i,j-nr] = ww_corner
                WW[i,j] =  ww_corner
                #j=j+1

    return M_l, WW
