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

import numpy as np


def dttv_ACC(DAT_Z, dt, nt, order):
    dttv3 = np.zeros((DAT_Z.shape))
   
    if order == 2:
        for i in range(1,nt-1):
            dttv3[:,i] = (DAT_Z[:,i-1] - (2*DAT_Z[:,i]) + DAT_Z[:,i+1])/ (dt**2)
            
    elif order == 4:
        for i in range(1,nt-2):
            dttv3[:,i] = ((-DAT_Z[:,i+2]) + (16*DAT_Z[:,i+1]) - (30*DAT_Z[:,i]) + (16*DAT_Z[:,i-1]) - DAT_Z[:,i-2])/ (12*(dt**2))
            
    elif order == 8:
        for i in range(1,nt-3):
            dttv3[:,i] = ((2*DAT_Z[:,i+3])+(-27*DAT_Z[:,i+2]) + (270*DAT_Z[:,i+1]) - (490*DAT_Z[:,i]) + (270*DAT_Z[:,i-1]) - (27*DAT_Z[:,i-2]) + (2*DAT_Z[:,i-3]))/ (180*(dt**2))
       
    return dttv3


def _1st_DER_ACC(FIELD_Z, dx, dy, dz, nrx, nry, nrz,order):
    ## Initialize Gradients
   
    FIELD_Z_gradX = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradY = np.zeros(FIELD_Z.shape)    
    FIELD_Z_gradZ = np.zeros(FIELD_Z.shape)
    
    if order == 2:
        for pos in range(1, nrx-1):
            FIELD_Z_gradX[pos,:,:,:] = (FIELD_Z[pos+1,:,:,:] - FIELD_Z[pos-1,:,:,:])/(2*dx)

        for pos in range(1, nry-1):
            FIELD_Z_gradY[:,pos,:,:] = (FIELD_Z[:,pos+1,:,:] - FIELD_Z[:,pos-1,:,:])/(2*dy)

        for pos in range(1, nrz-1):
            FIELD_Z_gradZ[:,:,pos,:] = (-FIELD_Z[:,:,pos+1,:] + FIELD_Z[:,:,pos-1,:])/(2*dz)
                
            
    if order == 4:
        for pos in range(2, nrx-2):
            FIELD_Z_gradX[pos,:,:,:] = (-FIELD_Z[pos+2,:,:,:] + 8*FIELD_Z[pos+1,:,:,:] - 8*FIELD_Z[pos-1,:,:,:] + FIELD_Z[pos-2,:,:,:])/(12*dx)

        for pos in range(2, nry-2):
            FIELD_Z_gradY[:,pos,:,:] = (-FIELD_Z[:,pos+2,:,:] + 8*FIELD_Z[:,pos+1,:,:] - 8*FIELD_Z[:,pos-1,:,:] + FIELD_Z[:,pos-2,:,:])/(12*dy)

        for pos in range(2, nrz-2):
            FIELD_Z_gradZ[:,:,pos,:] = (-FIELD_Z[:,:,pos+2,:] + 8*FIELD_Z[:,:,pos+1,:] - 8*FIELD_Z[:,:,pos-1,:] + FIELD_Z[:,:,pos-2,:])/(-12*dz)
        
        
    return FIELD_Z_gradX, FIELD_Z_gradY, FIELD_Z_gradZ


def _2nd_DER_ACC(FIELD_Z, dx, dy, dz, nrx, nry, nrz, order):
    ## Initialize Gradients
   
    FIELD_Z_gradXX = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradYY = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradZZ = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradXY = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradYX = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradYZ = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradXZ = np.zeros(FIELD_Z.shape)
    
    # 2nd order accurate
    if order == 2:
        for pos in range(1, nrx-1):
                FIELD_Z_gradXX[pos,:,:,:] = (FIELD_Z[pos+1,:,:,:] - (2* FIELD_Z[pos,:,:,:]) + FIELD_Z[pos-1,:,:,:])/(dx**2)

        for pos in range(1, nry-1):
                FIELD_Z_gradYY[:,pos,:,:] = (FIELD_Z[:,pos+1,:,:] -  (2* FIELD_Z[:,pos,:,:]) +  FIELD_Z[:,pos-1,:,:])/(dy**2)

        for pos in range(1, nrz-1):
               FIELD_Z_gradZZ[:,:,pos,:] = (FIELD_Z[:,:,pos+1,:] - (2* FIELD_Z[:,:,pos,:]) + FIELD_Z[:,:,pos-1,:])/(dz**2)


        for posx in range(1, nrx-1):
            for posy in range(1, nry-1):
                FIELD_Z_gradXY[posx,posy,:,:] = (FIELD_Z[posx+1,posy+1,:,:] - FIELD_Z[posx-1,posy+1,:,:] - \
                                            FIELD_Z[posx+1,posy-1,:,:] + FIELD_Z[posx-1,posy-1,:,:])/(4*dx*dy)

        FIELD_Z_gradYX = FIELD_Z_gradXY


        for posz in range(1, nrz-1):
            for posy in range(1, nry-1):
                FIELD_Z_gradYZ[:,posy,posz,:] = (FIELD_Z[:,posy+1,posz+1,:] - FIELD_Z[:,posy+1,posz-1,:] - \
                                                    FIELD_Z[:,posy-1,posz+1,:] + FIELD_Z[:,posy-1,posz-1,:])/-(4*dz*dy)

        for posz in range(1, nrz-1):
            for posx in range(1, nrx-1):
                FIELD_Z_gradXZ[posx,:,posz,:] = (FIELD_Z[posx+1,:,posz+1,:] - FIELD_Z[posx+1,:,posz-1,:] - \
                                                    FIELD_Z[posx-1,:,posz+1,:] + FIELD_Z[posx-1,:,posz-1,:])/-(4*dz*dx)

    if order == 4:
        for pos in range(2, nrx-2):
                FIELD_Z_gradXX[pos,:,:,:] = (-FIELD_Z[pos+2,:,:,:] + 16*FIELD_Z[pos+1,:,:,:] - 30*FIELD_Z[pos,:,:,:] \
                                             + 16*FIELD_Z[pos-1,:,:,:] -  FIELD_Z[pos-2,:,:,:] )/(12*(dx**2))

        for pos in range(2, nry-2):
                FIELD_Z_gradYY[:,pos,:,:] = (-FIELD_Z[:,pos+2,:,:] + 16*FIELD_Z[:,pos+1,:,:] - 30*FIELD_Z[:,pos,:,:] \
                                             + 16*FIELD_Z[:,pos-1,:,:] -  FIELD_Z[:,pos-2,:,:] )/(12*(dy**2))


        for pos in range(2, nrz-2):
                FIELD_Z_gradZZ[:,:,pos,:] = (-FIELD_Z[:,:,pos+2,:] + 16*FIELD_Z[:,:,pos+1,:] - 30*FIELD_Z[:,:,pos,:] \
                                             + 16*FIELD_Z[:,:,pos-1,:] -  FIELD_Z[:,:,pos-2,:] )/(12*(dz**2))
                



        for posx in range(2, nrx-2):
            for posy in range(2, nry-2):
                FIELD_Z_gradXY[posx,posy,:,:] = (8*(FIELD_Z[posx+1,posy-2,:,:]+FIELD_Z[posx+2,posy-1,:,:]+FIELD_Z[posx-2,posy+1,:,:]+FIELD_Z[posx-1,posy+2,:,:])\
                                                - 8*(FIELD_Z[posx-1,posy-2,:,:]+FIELD_Z[posx-2,posy-1,:,:]+FIELD_Z[posx+1,posy+2,:,:]+FIELD_Z[posx+2,posy+1,:,:])\
                                                - (FIELD_Z[posx+2,posy-2,:,:]+FIELD_Z[posx-2,posy+2,:,:]-FIELD_Z[posx-2,posy-2,:,:]-FIELD_Z[posx+2,posy+2,:,:])\
                                                + 64 * (FIELD_Z[posx-1,posy-1,:,:]+FIELD_Z[posx+1,posy+1,:,:]-FIELD_Z[posx+1,posy-1,:,:]-FIELD_Z[posx-1,posy+1,:,:]))/(144*dx*dy)

     
        FIELD_Z_gradYX = FIELD_Z_gradXY



        for posz in range(2, nrz-2):
            for posy in range(2, nry-2):
                FIELD_Z_gradYZ[:,posy,posz,:] = (8*(FIELD_Z[:,posy+1,posz-2,:]+FIELD_Z[:,posy+2,posz-1,:]+FIELD_Z[:,posy-2,posz+1,:]+FIELD_Z[:,posy-1,posz+2,:])\
                                                - 8*(FIELD_Z[:,posy-1,posz-2,:]+FIELD_Z[:,posy-2,posz-1,:]+FIELD_Z[:,posy+1,posz+2,:]+FIELD_Z[:,posy+2,posz+1,:])\
                                                - (FIELD_Z[:,posy+2,posz-2,:]+FIELD_Z[:,posy-2,posz+2,:]-FIELD_Z[:,posy-2,posz-2,:]-FIELD_Z[:,posy+2,posz+2,:])\
                                                + 64 * (FIELD_Z[:,posy-1,posz-1,:]+FIELD_Z[:,posy+1,posz+1,:]-FIELD_Z[:,posy+1,posz-1,:]-FIELD_Z[:,posy-1,posz+1,:]))/-(144*dz*dy)


        for posz in range(2, nrz-2):
            for posx in range(2, nrx-2):
                FIELD_Z_gradXZ[posx,:,posz,:] = (8*(FIELD_Z[posx+1,:,posz-2,:]+FIELD_Z[posx+2,:,posz-1,:]+FIELD_Z[posx-2,:,posz+1,:]+FIELD_Z[posx-1,:,posz+2,:])\
                                                - 8*(FIELD_Z[posx-1,:,posz-2,:]+FIELD_Z[posx-2,:,posz-1,:]+FIELD_Z[posx+1,:,posz+2,:]+FIELD_Z[posx+2,:,posz+1,:])\
                                                - (FIELD_Z[posx+2,:,posz-2,:]+FIELD_Z[posx-2,:,posz+2,:]-FIELD_Z[posx-2,:,posz-2,:]-FIELD_Z[posx+2,:,posz+2,:])\
                                                + 64 * (FIELD_Z[posx-1,:,posz-1,:]+FIELD_Z[posx+1,:,posz+1,:]-FIELD_Z[posx+1,:,posz-1,:]-FIELD_Z[posx-1,:,posz+1,:]))/-(144*dz*dx)

        
    return FIELD_Z_gradXX, FIELD_Z_gradYY, FIELD_Z_gradZZ,  FIELD_Z_gradXY, FIELD_Z_gradYX, FIELD_Z_gradYZ, FIELD_Z_gradXZ




