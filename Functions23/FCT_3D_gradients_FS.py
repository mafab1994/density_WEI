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


def FS_VERT_1stder_atFS(alpha, beta, FIELD_X_gradX, FIELD_Y_gradY, FIELD_Z_gradY, FIELD_Z_gradX):
    "First-Order Vertical Derivatives at the Free Surface above central point"
    
    FIELD_Z_gradZ_FreeSurf = - ( (alpha[:,:, np.newaxis]**2 - (2* (beta[:,:, np.newaxis]**2)) )\
                                / (alpha[:,:, np.newaxis]**2) ) * (FIELD_X_gradX + FIELD_Y_gradY)
    FIELD_Y_gradZ_FreeSurf = - FIELD_Z_gradY
    FIELD_X_gradZ_FreeSurf = - FIELD_Z_gradX
 
    return FIELD_X_gradZ_FreeSurf, FIELD_Y_gradZ_FreeSurf, FIELD_Z_gradZ_FreeSurf


def FD_VERT_1stder_notFS(FIELD_X_buried, FIELD_Y_buried, FIELD_Z_buried, FIELD_X_surf, FIELD_Y_surf, FIELD_Z_surf, dz):
    " Vertical derivative centered half the buried distance (dz/2) below the surface central point"
    
    FIELD_X_gradZ_FDnotFS = (-FIELD_X_buried + FIELD_X_surf)/dz
    FIELD_Y_gradZ_FDnotFS = (-FIELD_Y_buried + FIELD_Y_surf)/dz
    FIELD_Z_gradZ_FDnotFS = (-FIELD_Z_buried + FIELD_Z_surf)/dz
    
    return FIELD_X_gradZ_FDnotFS, FIELD_Y_gradZ_FDnotFS, FIELD_Z_gradZ_FDnotFS


def FD_VERT_2ndder_notFS(FIELD_X_gradZ_FDnotFS, FIELD_Y_gradZ_FDnotFS, FIELD_Z_gradZ_FDnotFS,\
                   FIELD_X_gradZ_FreeSurf, FIELD_Y_gradZ_FreeSurf, FIELD_Z_gradZ_FreeSurf, dz):
    "2nd order vertical derivative, centered at dx3/4 below the central surface point"
    
    FIELD_X_gradZZ_FDnotFS = 2* (FIELD_X_gradZ_FDnotFS - FIELD_X_gradZ_FreeSurf) / dz
    FIELD_Y_gradZZ_FDnotFS = 2* (FIELD_Y_gradZ_FDnotFS - FIELD_Y_gradZ_FreeSurf) / dz
    FIELD_Z_gradZZ_FDnotFS = 2* (FIELD_Z_gradZ_FDnotFS - FIELD_Z_gradZ_FreeSurf) / dz

    return FIELD_X_gradZZ_FDnotFS, FIELD_Y_gradZZ_FDnotFS, FIELD_Z_gradZZ_FDnotFS


# +
def mixed_FD_VERT_2nd_notFS(FIELD_X_buried, FIELD_Y_buried, FIELD_Z_buried,FIELD_X_2buried, FIELD_Y_2buried, FIELD_Z_2buried,FIELD_X_3buried, FIELD_Y_3buried, FIELD_Z_3buried, FIELD_X_surf, FIELD_Y_surf, FIELD_Z_surf, dz,dy,dx,nry,nrx):
    
    FIELD_X_gradZZ = np.zeros(FIELD_X_buried.shape)
    FIELD_Y_gradZZ = np.zeros(FIELD_X_buried.shape)
    FIELD_Z_gradZZ = np.zeros(FIELD_X_buried.shape)
    FIELD_X_gradYZ = np.zeros(FIELD_X_buried.shape)
    FIELD_Y_gradYZ = np.zeros(FIELD_X_buried.shape)
    FIELD_Z_gradYZ = np.zeros(FIELD_X_buried.shape)
    FIELD_X_gradXZ = np.zeros(FIELD_X_buried.shape)
    FIELD_Y_gradXZ = np.zeros(FIELD_X_buried.shape)
    FIELD_Z_gradXZ = np.zeros(FIELD_X_buried.shape)
    
   
#     FIELD_X_gradZZ[:,:,:] = (2*FIELD_X_surf[:,:,:] - (5*FIELD_X_buried[:,:,:]) + (4*FIELD_X_2buried[:,:,:])-  (FIELD_X_3buried[:,:,:]))/(dz**3)
#     FIELD_Y_gradZZ[:,:,:] = (2*FIELD_Y_surf[:,:,:] - (5*FIELD_Y_buried[:,:,:]) + (4*FIELD_Y_2buried[:,:,:])-  (FIELD_Y_3buried[:,:,:]))/(dz**3)
#     FIELD_Z_gradZZ[:,:,:] = (2*FIELD_Z_surf[:,:,:] - (5*FIELD_Z_buried[:,:,:]) + (4*FIELD_Z_2buried[:,:,:])-  (FIELD_Z_3buried[:,:,:]))/(dz**3)
    FIELD_X_gradZZ[:,:,:] = (FIELD_X_surf[:,:,:] - (2*FIELD_X_buried[:,:,:]) + (FIELD_X_2buried[:,:,:]))/(dz**2)
    FIELD_Y_gradZZ[:,:,:] = (FIELD_Y_surf[:,:,:] - (2*FIELD_Y_buried[:,:,:]) + (FIELD_Y_2buried[:,:,:]))/(dz**2)
    FIELD_Z_gradZZ[:,:,:] = (FIELD_Z_surf[:,:,:] - (2*FIELD_Z_buried[:,:,:]) + (FIELD_Z_2buried[:,:,:]))/(dz**2)
                            
    for posy in range(1, nry-1):
                    FIELD_X_gradYZ[:,posy,:] = (FIELD_X_buried[:,posy+1,:] - FIELD_X_surf[:,posy+1,:] - \
                                                        FIELD_X_buried[:,posy-1,:] + FIELD_X_surf[:,posy-1,:])/-(2*dz*dy)
                    FIELD_Y_gradYZ[:,posy,:] = (FIELD_Y_buried[:,posy+1,:] - FIELD_Y_surf[:,posy+1,:] - \
                                                        FIELD_Y_buried[:,posy-1,:] + FIELD_Y_surf[:,posy-1,:])/-(2*dz*dy)
                    FIELD_Z_gradYZ[:,posy,:] = (FIELD_Z_buried[:,posy+1,:] - FIELD_Z_surf[:,posy+1,:] - \
                                                        FIELD_Z_buried[:,posy-1,:] + FIELD_Z_surf[:,posy-1,:])/-(2*dz*dy)
    for posx in range(1, nrx-1):
                    FIELD_X_gradXZ[posx,:,:] = (FIELD_X_buried[posx+1,:,:] - FIELD_X_surf[posx+1,:,:] - \
                                                        FIELD_X_buried[posx-1,:,:] + FIELD_X_surf[posx-1,:,:])/-(2*dz*dx)
                    FIELD_Y_gradXZ[posx,:,:] = (FIELD_Y_buried[posx+1,:,:] - FIELD_Y_surf[posx+1,:,:] - \
                                                        FIELD_Y_buried[posx-1,:,:] + FIELD_Y_surf[posx-1,:,:])/-(2*dz*dx)
                    FIELD_Z_gradXZ[posx,:,:] = (FIELD_Z_buried[posx+1,:,:] - FIELD_Z_surf[posx+1,:,:] - \
                                                        FIELD_Z_buried[posx-1,:,:] + FIELD_Z_surf[posx-1,:,:])/-(2*dz*dx)
                
    return FIELD_X_gradZZ,FIELD_Y_gradZZ,FIELD_Z_gradZZ,FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ


# -

def FS_VERT_2ndder_atFS(dttv1, dttv2, dttv3, alpha, beta, FIELD_Z_gradXX, FIELD_Z_gradYY, FIELD_Y_gradXX, FIELD_X_gradXX,\
                        FIELD_Y_gradXY, FIELD_X_gradYX, FIELD_Y_gradYY, FIELD_X_gradYY):
    "2nd order vertical derivative estimated at the free surface at the central point"
    
    FIELD_X_gradZZ_FreeSurf = (dttv1/beta[:,:, np.newaxis]**2) - (FIELD_X_gradXX + FIELD_X_gradYY) - \
                              (2*(1- (beta[:,:, np.newaxis]**2/alpha[:,:, np.newaxis]**2))  * (FIELD_X_gradXX + FIELD_Y_gradXY))
    
    FIELD_Y_gradZZ_FreeSurf = (dttv2/beta[:,:, np.newaxis]**2) - (FIELD_Y_gradXX + FIELD_Y_gradYY) \
                                - (2*(1- (beta[:,:, np.newaxis]**2/alpha[:,:, np.newaxis]**2)) * (FIELD_X_gradYX + FIELD_Y_gradYY))
    
    FIELD_Z_gradZZ_FreeSurf = (dttv3/alpha[:,:, np.newaxis]**2) + ((1-(2*(beta[:,:, np.newaxis]**2/alpha[:,:, np.newaxis]**2)))\
                                                                   * (FIELD_Z_gradXX + FIELD_Z_gradYY))
    
    return FIELD_X_gradZZ_FreeSurf, FIELD_Y_gradZZ_FreeSurf, FIELD_Z_gradZZ_FreeSurf


def LAX(dz, FIELD_X_gradZZ_FreeSurf, FIELD_Y_gradZZ_FreeSurf, FIELD_Z_gradZZ_FreeSurf):
    "Lax correction - from first order FD derivative to free surface"
    
    L11 = - (dz/2) * FIELD_X_gradZZ_FreeSurf
    L12 = - (dz/2) * FIELD_Y_gradZZ_FreeSurf
    L13 = - (dz/2) * FIELD_Z_gradZZ_FreeSurf
   
    return L11, L12, L13


def FD_VERT_1stder_LAX(L11, L12, L13, FIELD_X_gradZ_FDnotFS, FIELD_Y_gradZ_FDnotFS, FIELD_Z_gradZ_FDnotFS):
    " First vertical FD derivative - Lax corrected to estimate surface response"
    
    FIELD_X_gradZ_LAX = FIELD_X_gradZ_FDnotFS + L11
    FIELD_Y_gradZ_LAX = FIELD_Y_gradZ_FDnotFS + L12
    FIELD_Z_gradZ_LAX = FIELD_Z_gradZ_FDnotFS + L13
   
    return FIELD_X_gradZ_LAX, FIELD_Y_gradZ_LAX, FIELD_Z_gradZ_LAX
