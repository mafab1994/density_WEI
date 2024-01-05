# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:31:23 2020

@author: s2110831
"""

import numpy as np
# Dxx = (-2*sparse.eye(n_rx,n_rz,0)+sparse.eye(n_rx,n_rz,1)+sparse.eye(n_rx,n_rz,-1))/dist_r**2
# Dxx = (Dxx).todense() 
# Dxx = Dxx 
# Dzz = np.transpose(Dxx)
# D_LP = LaplacianMatrix(n_rx-2, n_rz-2, dist_r, dist_r).todense()

def LaplacianMatrix(nx,ny,dx,dy):
    """ compute del_squared  on nx by ny grid. Parameters are:
    nx -- number of points in x direction.
    ny -- number of points in y direction.
    dx -- separation between points in x direction.
    dy -- separation between points in y direction.""" 
    from scipy.sparse import eye
    
    N = nx*ny
    
    ## 1. Discretise to deal with the Internal Nodes
    dsqr = - ( 2/(dx**2) + 2/(dy**2) ) * eye(N,N,0)  # center point
    dsqr = dsqr + ( eye(N,N,1) + eye(N,N,-1) )/dx**2    # dx
    dsqr = dsqr + ( eye(N,N,nx) + eye(N,N,-nx) )/dy**2  # dy
    
    # ## 2. Reset the Boundary Nodes to the identity matrix
    #index_array=np.arange(ny*nx).reshape((ny,nx)) # array of indices
    # dsqr=dsqr.tolil() 
    
    # for row in range(nx):
    #     dsqr[row,row] = 1
    #     dsqr[row, row-1] = 0
    #     dsqr[row, row+1] = 0
    #     dsqr[row, row+nx] = 0
    #     dsqr[row, row-nx] = 0
    
    # for row in range(N-nx,N):
    #     dsqr[row,row] = 1
    #     dsqr[row, row-1] = 0
    #     if(row+1<N): 
    #         dsqr[row, row+1] = 0
    #     if(row+nx<N): 
    #         dsqr[row, row+nx] = 0
    #     dsqr[row, row-nx] = 0

    # for i in range(nx-1):
    #     row = nx*(i+1)
    #     dsqr[row,row] = 1
    #     dsqr[row, row-1] = 0
    #     if(row+1<N):
    #         dsqr[row, row+1] = 0
    #     if(row+nx<N): 
    #         dsqr[row, row+nx] = 0
    #     dsqr[row, row-nx] = 0
    
    # for i in range(nx-1):
    #     row = nx*(i+1)-1
    #     dsqr[row,row] = 1
    #     dsqr[row, row-1] = 0
    #     if(row+1<N):
    #         dsqr[row, row+1] = 0
    #     if(row+nx<N): 
    #         dsqr[row, row+nx] = 0
    #     dsqr[row, row-nx] = 0
    
    # dsqr=dsqr.tocsr() # convert back to efficient form
    
    return dsqr


def LaplacianMatrix_CST(nx,ny,dx,dy):
    """ compute del_squared  on nx by ny grid. Parameters are:
    nx -- number of points in x direction.
    ny -- number of points in y direction.
    dx -- separation between points in x direction.
    dy -- separation between points in y direction.""" 
    from scipy.sparse import eye
    N = nx*ny
    dsqr       = np.zeros((N,N))
    vR = np.arange(nx, N-nx, nx)
    vL = np.arange((2*nx)-1, N-nx, nx)
  
    
    for ii in range(0,1): 
        dsqr[ii,ii] = -(4)* 1/((dx**2))   # center point
        dsqr[ii,ii+1] = 1 /((dx**2))    # dx
        dsqr[ii,ii+nx] = 1/((dx**2))  # dy
        
        
    for ii in range(1,nx-1): 
        dsqr[ii,ii-1] = 1 /((dx**2))
        dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
        dsqr[ii,ii+1] = 1 /((dx**2))    # dx
        dsqr[ii,ii+nx] = 1/((dx**2))  # dy
        
    
    for ii in range(nx-1, nx):
        dsqr[ii,ii-1] = 1 /((dx**2))
        dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
        
        dsqr[ii,ii+nx] = 1/((dx**2))  # dy
    
   
    for ii in range(nx, N-nx): 
            if np.any(ii == vL[:])==True:
                dsqr[ii,ii-nx] = 1 /((dx**2))
                dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
                dsqr[ii,ii+1] = 1 /((dx**2))    # dx
                dsqr[ii,ii+nx] = 1/((dx**2))  # dy
                  

            elif np.any(ii == vR[:])==True:
                dsqr[ii,ii-nx] = 1 /((dx**2))
                dsqr[ii,ii-1] = 1 /((dx**2))    # dx
                dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
                dsqr[ii,ii+nx] = 1/((dx**2))  # dy
                 

            else:
                dsqr[ii,ii-nx] = 1 /((dx**2))
                dsqr[ii,ii-1] = 1 /((dx**2))    # dx
                dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
                dsqr[ii,ii+1] = 1/((dx**2))  # dy
                dsqr[ii,ii+nx] = 1/((dx**2))  # dy
                 
                
    for ii in range(N-nx, N-nx+1):
        dsqr[ii,ii-nx] = 1 /((dx**2))
                
        dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
        dsqr[ii,ii+1] = 1/((dx**2))  # dy
                
        
                    
    for ii in range(N-nx+1, N-1):                     
                             
        dsqr[ii,ii-nx] = 1 /((dx**2))
        dsqr[ii,ii-1] = 1 /((dx**2))    # dx
        dsqr[ii,ii] = -(4)* 1/((dx**2))   # center point
        dsqr[ii,ii+1] = 1/((dx**2))  # dy
        
        
    
    for ii in range(N-1, N):
        
        dsqr[ii,ii-nx] = 1 /((dx**2))
        dsqr[ii,ii-1] = 1 /((dx**2))    # dx
        dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
                 
    
        
        
    return dsqr


def LaplacianMatrix_CST_4th(nx,ny,dx,dy):
    """ compute del_squared  on nx by ny grid. Parameters are:
    nx -- number of points in x direction.
    ny -- number of points in y direction.
    dx -- separation between points in x direction.
    dy -- separation between points in y direction.""" 
    from scipy.sparse import eye
    N = nx*ny
    dsqr       = np.zeros((N,N))
    vR = np.arange(nx, N-nx, nx)
    vL = np.arange((2*nx)-1, N-nx, nx)
  
    
    for ii in range(0,1): 
        dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
        dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
        dsqr[ii,ii+2] = (-1/12)  /((dx**2))    # dx
        dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
        dsqr[ii,ii+nx+1] = (-1/12)  /((dx**2))  # dy
        
        
    for ii in range(1,2): 
        dsqr[ii,ii-1] = (4/3)  /((dx**2))
        dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
        dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
        dsqr[ii,ii+2] = (-1/12)  /((dx**2))    # dx
        dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
        dsqr[ii,ii+(2*nx)] = (-1/12)  /((dx**2))  # dy
        
    for ii in range(2,nx-2): 
        dsqr[ii,ii-2] = (-1/12)  /((dx**2))
        dsqr[ii,ii-1] = (4/3)  /((dx**2))
        dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
        dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
        dsqr[ii,ii+2] = (-1/12)  /((dx**2))    # dx
        dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
        dsqr[ii,ii+(2*nx)] = (-1/12)  /((dx**2))  # dy
        
    for ii in range(nx-2, nx-1): 
        dsqr[ii,ii-2] = (-1/12)  /((dx**2))
        dsqr[ii,ii-1] = (4/3)  /((dx**2))
        dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
        dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
        #dsqr[ii,ii+2] = (-1/12) * /((dx**2))    # dx
        dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
        dsqr[ii,ii+(2*nx)] = (-1/12)  /((dx**2))  # dy
        
    for ii in range(nx-1, nx):
        dsqr[ii,ii-2] = (-1/12)  /((dx**2))
        dsqr[ii,ii-1] = (4/3)  /((dx**2))
        dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
        dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
        dsqr[ii,ii+(2*nx)] = (-1/12)  /((dx**2))  # dy
   
    for ii in range(nx, 2*nx): 
            if np.any(ii == vL[:])==True:
                dsqr[ii,ii-nx] = (4/3)  /((dx**2))
                dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
                dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
                dsqr[ii,ii+2] = (-1/12)  /((dx**2))    # dx
                dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
                  

            elif np.any(ii == vR[:])==True:
                dsqr[ii,ii-nx] = (4/3)  /((dx**2))
                dsqr[ii,ii-2] = (-1/12)  /((dx**2))    # dx
                dsqr[ii,ii-1] = (4/3)  /((dx**2))
                dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
                dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
                 

            else:
                dsqr[ii,ii-nx] = (4/3)  /((dx**2))
                dsqr[ii,ii-2] = (-1/12)  /((dx**2))    # dx
                dsqr[ii,ii-1] = (4/3)  /((dx**2))
                dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
                dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
                dsqr[ii,ii+2] = (-1/12)  /((dx**2))    # dx
                dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
                
    for ii in range(2*nx, N-nx): 
            if np.any(ii == vL[:])==True:
                dsqr[ii,ii-(2*nx)] = (-1/12)  /((dx**2))
                dsqr[ii,ii-nx] = (4/3)  /((dx**2))
                dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
                dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
                dsqr[ii,ii+2] = (-1/12)  /((dx**2))    # dx
                dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
                  

            elif np.any(ii == vR[:])==True:
                dsqr[ii,ii-(2*nx)] = (-1/12)  /((dx**2))
                dsqr[ii,ii-nx] = (4/3)  /((dx**2))
                dsqr[ii,ii-2] = (-1/12)  /((dx**2))    # dx
                dsqr[ii,ii-1] = (4/3)  /((dx**2))
                dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
                dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
                 

            else:
                dsqr[ii,ii-(2*nx)] = (-1/12)  /((dx**2))
                dsqr[ii,ii-nx] = (4/3)  /((dx**2))
                dsqr[ii,ii-2] = (-1/12)  /((dx**2))    # dx
                dsqr[ii,ii-1] = (4/3)  /((dx**2))
                dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
                dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
                dsqr[ii,ii+2] = (-1/12)  /((dx**2))    # dx
                dsqr[ii,ii+nx] = (4/3)  /((dx**2))  # dy
                 
                
    for ii in range(N-nx, N-nx+1):
        dsqr[ii,ii-(2*nx)] = (-1/12)  /((dx**2))
        dsqr[ii,ii-nx] = (4/3)  /((dx**2))        
        dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
        dsqr[ii,ii+1] = (4/3)  /((dx**2))    # dx
                
        
                    
    for ii in range(N-nx+1, N-1):                     
        dsqr[ii,ii-(2*nx)] = (-1/12) /((dx**2))                     
        dsqr[ii,ii-nx] = (4/3) /((dx**2))
        dsqr[ii,ii-1] = (4/3) /((dx**2))
        dsqr[ii,ii] = -(10/2) /((dx**2))   # center point
        dsqr[ii,ii+1] = (4/3) /((dx**2))    # dx
        
        
    
    for ii in range(N-1, N):
        dsqr[ii,ii-(2*nx)] = (-1/12) /((dx**2))
        dsqr[ii,ii-nx] = (4/3) /((dx**2))
        dsqr[ii,ii-1] = (4/3) /((dx**2))
        dsqr[ii,ii] = -(10/2)/((dx**2))   # center point
                 
    
        
        
    return dsqr


def LaplacianMatrix_VAR(nx,ny,dx,dy, rho):
    """ compute del_squared  on nx by ny grid. Parameters are:
    nx -- number of points in x direction.
    ny -- number of points in y direction.
    dx -- separation between points in x direction.
    dy -- separation between points in y direction.""" 
    
    N = nx*ny
    dsqr       = np.zeros((N,N))
    vR = np.arange(nx, N-nx, nx)
    vL = np.arange((2*nx)-1, N-nx, nx)
    
    
    rho = rho.flatten()

  
    
    for ii in range(0,1): 
        dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii+1])+ (rho[ii]/rho[ii+nx]) )/(2*(dx**2))   # center point
        dsqr[ii,ii+1] = (1+ rho[ii]/rho[ii+1])/(2*(dx**2))    # dx
        dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii+nx])/(2*(dx**2))  # dy
        
        
    for ii in range(1,nx-1): 
        dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii-1]) /(2*(dx**2))
        dsqr[ii,ii] = -(4+  (rho[ii]/rho[ii-1]) + (rho[ii]/rho[ii+1]) + (rho[ii]/rho[ii+nx]))/(2*(dx**2))    # center point
        dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii+1]) /(2*(dx**2))    # dx
        dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii+nx])/(2*(dx**2))  # dy
        
    
    for ii in range(nx-1, nx):
        dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii-1]) /(2*(dx**2))
        dsqr[ii,ii] = -(4+  (rho[ii]/rho[ii-1])  + (rho[ii]/rho[ii+nx]))/(2*(dx**2))    # center point
        
        dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii+nx])/(2*(dx**2))  # dy
    
   
    for ii in range(nx, N-nx): 
            if np.any(ii == vL[:])==True:
                dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii-nx]) /(2*(dx**2))
                dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii-nx]) + (rho[ii]/rho[ii+nx]) + (rho[ii]/rho[ii+1]))/(2*(dx**2))    # center point
                dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii+1]) /(2*(dx**2))    # dx
                dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii+nx])/(2*(dx**2))  # dy
                  

            elif np.any(ii == vR[:])==True:
                dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii-nx]) /(2*(dx**2))
                dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii-1]) /(2*(dx**2))    # dx
                dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii-nx]) + (rho[ii]/rho[ii-1]) + (rho[ii]/rho[ii+nx]))/(2*(dx**2))    # center point
                dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii+nx])/(2*(dx**2))  # dy
                 

            else:
                dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii-nx])/(2*(dx**2))
                dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii-1]) /(2*(dx**2))    # dx
                dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii-nx]) + (rho[ii]/rho[ii-1]) + (rho[ii]/rho[ii+1])+ (rho[ii]/rho[ii+nx]) )/(2*(dx**2))    # center point
                dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii+1])/(2*(dx**2))  # dy
                dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii+nx])/(2*(dx**2))  # dy
                 
                
    for ii in range(N-nx, N-nx+1):
        dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii-nx]) /(2*(dx**2))
                
        dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii-nx])  + (rho[ii]/rho[ii+1]))/(2*(dx**2))    # center point
        dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii+1])/(2*(dx**2))  # dy
                
        
                    
    for ii in range(N-nx+1, N-1):                     
                             
        dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii-nx]) /(2*(dx**2))
        dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii-1]) /(2*(dx**2))    # dx
        dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii-nx]) + (rho[ii]/rho[ii-1]) + (rho[ii]/rho[ii+1]))/(2*(dx**2))   # center point
        dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii+1])/(2*(dx**2))  # dy
        
        
    
    for ii in range(N-1, N):
        
        dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii-nx]) /(2*(dx**2))
        dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii-1]) /(2*(dx**2))    # dx
        dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii-nx]) + (rho[ii]/rho[ii-1]) )/(2*(dx**2))    # center point
                 
    
        
        
        
        
    return dsqr


def LaplacianMatrix_VAR_ELASTIC(nx,ny,dx,dy, rho):
    """ compute del_squared  on nx by ny grid. Parameters are:
    nx -- number of points in x direction.
    ny -- number of points in y direction.
    dx -- separation between points in x direction.
    dy -- separation between points in y direction.""" 
    
    N = nx*ny
    dsqr       = np.zeros((N,N))
    vR = np.arange(nx, N-nx, nx)
    vL = np.arange((2*nx)-1, N-nx, nx)
    
    
    rho = rho.flatten()

  
    
    for ii in range(0,1): 
        dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii-1])+ (rho[ii]/rho[ii-nx]) )/(2*(dx**2))   # center point
        dsqr[ii,ii+1] = (1+ rho[ii]/rho[ii-1])/(2*(dx**2))    # dx
        dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii-nx])/(2*(dx**2))  # dy
        
        
    for ii in range(1,nx-1): 
        dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii+1]) /(2*(dx**2))
        dsqr[ii,ii] = -(4+  (rho[ii]/rho[ii+1]) + (rho[ii]/rho[ii-1]) + (rho[ii]/rho[ii-nx]))/(2*(dx**2))    # center point
        dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii-1]) /(2*(dx**2))    # dx
        dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii-nx])/(2*(dx**2))  # dy
        
    
    for ii in range(nx-1, nx):
        dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii+1]) /(2*(dx**2))
        dsqr[ii,ii] = -(4+  (rho[ii]/rho[ii+1])  + (rho[ii]/rho[ii-nx]))/(2*(dx**2))    # center point
        
        dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii-nx])/(2*(dx**2))  # dy
    
   
    for ii in range(nx, N-nx): 
            if np.any(ii == vL[:])==True:
                dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii+nx]) /(2*(dx**2))
                dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii+nx]) + (rho[ii]/rho[ii-nx]) + (rho[ii]/rho[ii-1]))/(2*(dx**2))    # center point
                dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii-1]) /(2*(dx**2))    # dx
                dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii-nx])/(2*(dx**2))  # dy
                  

            elif np.any(ii == vR[:])==True:
                dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii+nx]) /(2*(dx**2))
                dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii+1]) /(2*(dx**2))    # dx
                dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii+nx]) + (rho[ii]/rho[ii+1]) + (rho[ii]/rho[ii-nx]))/(2*(dx**2))    # center point
                dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii-nx])/(2*(dx**2))  # dy
                 

            else:
                dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii+nx])/(2*(dx**2))
                dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii+1]) /(2*(dx**2))    # dx
                dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii+nx]) + (rho[ii]/rho[ii+1]) + (rho[ii]/rho[ii-1])+ (rho[ii]/rho[ii-nx]) )/(2*(dx**2))    # center point
                dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii-1])/(2*(dx**2))  # dy
                dsqr[ii,ii+nx] = (1 + rho[ii]/rho[ii-nx])/(2*(dx**2))  # dy
                 
                
    for ii in range(N-nx, N-nx+1):
        dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii+nx]) /(2*(dx**2))
                
        dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii+nx])  + (rho[ii]/rho[ii-1]))/(2*(dx**2))    # center point
        dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii-1])/(2*(dx**2))  # dy
                
        
                    
    for ii in range(N-nx+1, N-1):                     
                             
        dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii+nx]) /(2*(dx**2))
        dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii+1]) /(2*(dx**2))    # dx
        dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii+nx]) + (rho[ii]/rho[ii+1]) + (rho[ii]/rho[ii-1]))/(2*(dx**2))   # center point
        dsqr[ii,ii+1] = (1 + rho[ii]/rho[ii-1])/(2*(dx**2))  # dy
        
        
    
    for ii in range(N-1, N):
        
        dsqr[ii,ii-nx] = (1 + rho[ii]/rho[ii+nx]) /(2*(dx**2))
        dsqr[ii,ii-1] = (1 + rho[ii]/rho[ii+1]) /(2*(dx**2))    # dx
        dsqr[ii,ii] = -(4+ (rho[ii]/rho[ii+nx]) + (rho[ii]/rho[ii+1]) )/(2*(dx**2))    # center point
                 
    
        
        
        
        
    return dsqr


def LaplacianMatrix_CST_3D(nx,ny,nz,dx,dy,dz):
    """ compute del_squared  on nx by ny grid. Parameters are:
    nx -- number of points in x direction.
    ny -- number of points in y direction.
    dx -- separation between points in x direction.
    dy -- separation between points in y direction.""" 
    from scipy.sparse import eye
    N = nx*ny*nz
    dsqr       = np.zeros((N,N))
    vR = np.arange(nx, N-nx, nx)
    vL = np.arange((2*nx)-1, N-nx, nx)
  
    
    for ii in range(0,1): 
        dsqr[ii,ii] = -(4)* 1/((dx**2))   # center point
        dsqr[ii,ii+1] = 1 /((dx**2))    # dx
        dsqr[ii,ii+nx] = 1/((dx**2))  # dy
        
        
    for ii in range(1,nx-1): 
        dsqr[ii,ii-1] = 1 /((dx**2))
        dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
        dsqr[ii,ii+1] = 1 /((dx**2))    # dx
        dsqr[ii,ii+nx] = 1/((dx**2))  # dy
        
    
    for ii in range(nx-1, nx):
        dsqr[ii,ii-1] = 1 /((dx**2))
        dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
        
        dsqr[ii,ii+nx] = 1/((dx**2))  # dy
    
   
    for ii in range(nx, N-nx): 
            if np.any(ii == vL[:])==True:
                dsqr[ii,ii-nx] = 1 /((dx**2))
                dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
                dsqr[ii,ii+1] = 1 /((dx**2))    # dx
                dsqr[ii,ii+nx] = 1/((dx**2))  # dy
                  

            elif np.any(ii == vR[:])==True:
                dsqr[ii,ii-nx] = 1 /((dx**2))
                dsqr[ii,ii-1] = 1 /((dx**2))    # dx
                dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
                dsqr[ii,ii+nx] = 1/((dx**2))  # dy
                 

            else:
                dsqr[ii,ii-nx] = 1 /((dx**2))
                dsqr[ii,ii-1] = 1 /((dx**2))    # dx
                dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
                dsqr[ii,ii+1] = 1/((dx**2))  # dy
                dsqr[ii,ii+nx] = 1/((dx**2))  # dy
                 
                
    for ii in range(N-nx, N-nx+1):
        dsqr[ii,ii-nx] = 1 /((dx**2))
                
        dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
        dsqr[ii,ii+1] = 1/((dx**2))  # dy
                
        
                    
    for ii in range(N-nx+1, N-1):                     
                             
        dsqr[ii,ii-nx] = 1 /((dx**2))
        dsqr[ii,ii-1] = 1 /((dx**2))    # dx
        dsqr[ii,ii] = -(4)* 1/((dx**2))   # center point
        dsqr[ii,ii+1] = 1/((dx**2))  # dy
        
        
    
    for ii in range(N-1, N):
        
        dsqr[ii,ii-nx] = 1 /((dx**2))
        dsqr[ii,ii-1] = 1 /((dx**2))    # dx
        dsqr[ii,ii] = -(4)* 1/((dx**2))    # center point
                 
    
        
        
    return dsqr_3D
