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


def dtv(DAT_X, DAT_Y, DAT_Z, dt, nt, order):
    dtv3 = np.zeros((DAT_X.shape))
    dtv2 = np.zeros((DAT_Y.shape))
    dtv1 = np.zeros((DAT_Z.shape))
    
    for i in range(1,nt-1):
            dtv3[:,i] = (DAT_Z[:,i+1] - DAT_Z[:,i-1])/ (dt*2)
            dtv2[:,i] = (DAT_Y[:,i+1] - DAT_Y[:,i-1])/ (dt*2)
            dtv1[:,i] = (DAT_X[:,i+1] - DAT_X[:,i-1])/ (dt*2)
            
    return dtv1, dtv2, dtv3


# +
def dttv(DAT_X, DAT_Y, DAT_Z, dt, nt, order):
    dttv3 = np.zeros((DAT_X.shape))
    dttv2 = np.zeros((DAT_Y.shape))
    dttv1 = np.zeros((DAT_Z.shape))
    
    if order == 2:
        for i in range(1,nt-1):
            dttv3[:,i] = (DAT_Z[:,i-1] - (2*DAT_Z[:,i]) + DAT_Z[:,i+1])/ (dt**2)
            dttv2[:,i] = (DAT_Y[:,i-1] - (2*DAT_Y[:,i]) + DAT_Y[:,i+1])/ (dt**2)
            dttv1[:,i] = (DAT_X[:,i-1] - (2*DAT_X[:,i]) + DAT_X[:,i+1])/ (dt**2)
       
        
    elif order == 4:
        for i in range(1,nt-2):
            dttv3[:,i] = ((-DAT_Z[:,i+2]) + (16*DAT_Z[:,i+1]) - (30*DAT_Z[:,i]) + (16*DAT_Z[:,i-1]) - DAT_Z[:,i-2])/ (12*(dt**2))
            dttv2[:,i] = ((-DAT_Y[:,i+2]) + (16*DAT_Y[:,i+1]) - (30*DAT_Y[:,i]) + (16*DAT_Y[:,i-1]) - DAT_Y[:,i-2])/ (12*(dt**2))
            dttv1[:,i] = ((-DAT_X[:,i+2]) + (16*DAT_X[:,i+1]) - (30*DAT_X[:,i]) + (16*DAT_X[:,i-1]) - DAT_X[:,i-2])/ (12*(dt**2))
        
    
    elif order == 8:
        for i in range(1,nt-3):
            dttv3[:,i] = ((2*DAT_Z[:,i+3])+(-27*DAT_Z[:,i+2]) + (270*DAT_Z[:,i+1]) - (490*DAT_Z[:,i]) + (270*DAT_Z[:,i-1]) - (27*DAT_Z[:,i-2]) + (2*DAT_Z[:,i-3]))/ (180*(dt**2))
            dttv2[:,i] = ((2*DAT_Y[:,i+3])+(-27*DAT_Y[:,i+2]) + (270*DAT_Y[:,i+1]) - (490*DAT_Y[:,i]) + (270*DAT_Y[:,i-1]) - (27*DAT_Y[:,i-2]) + (2*DAT_Y[:,i-3]))/ (180*(dt**2))
            dttv1[:,i] = ((2*DAT_X[:,i+3])+(-27*DAT_X[:,i+2]) + (270*DAT_X[:,i+1]) - (490*DAT_X[:,i]) + (270*DAT_X[:,i-1]) - (27*DAT_X[:,i-2]) + (2*DAT_X[:,i-3]))/ (180*(dt**2))
    
#     if order == 2:
#         dttv3  = np.asarray([(DAT_Z[:,i-1] - (2*DAT_Z[:,i]) + DAT_Z[:,i+1])/ (dt**2) for i in range(1,nt-1)])
#         dttv2  = np.asarray([(DAT_Y[:,i-1] - (2*DAT_Y[:,i]) + DAT_Y[:,i+1])/ (dt**2) for i in range(1,nt-1)])
#         dttv1 = np.asarray([(DAT_X[:,i-1] - (2*DAT_X[:,i]) + DAT_X[:,i+1])/ (dt**2) for i in range(1,nt-1)])
       
        
#     elif order == 4:
#         for i in range(1,nt-2):
#             dttv3 = np.asarray([((-DAT_Z[:,i+2]) + (16*DAT_Z[:,i+1]) - (30*DAT_Z[:,i]) + (16*DAT_Z[:,i-1]) - DAT_Z[:,i-2])/ (12*(dt**2)) for i in range(1,nt-1)])
#             dttv2 = np.asarray([((-DAT_Y[:,i+2]) + (16*DAT_Y[:,i+1]) - (30*DAT_Y[:,i]) + (16*DAT_Y[:,i-1]) - DAT_Y[:,i-2])/ (12*(dt**2)) for i in range(1,nt-1)])
#             dttv1 = np.asarray([((-DAT_X[:,i+2]) + (16*DAT_X[:,i+1]) - (30*DAT_X[:,i]) + (16*DAT_X[:,i-1]) - DAT_X[:,i-2])/ (12*(dt**2)) for i in range(1,nt-1)])
        
    
#     elif order == 8:
#         for i in range(1,nt-3):
#             dttv3 = np.asarray([((2*DAT_Z[:,i+3])+(-27*DAT_Z[:,i+2]) + (270*DAT_Z[:,i+1]) - (490*DAT_Z[:,i]) + (270*DAT_Z[:,i-1]) - (27*DAT_Z[:,i-2]) + (2*DAT_Z[:,i-3]))/ (180*(dt**2)) for i in range(1,nt-1)])
#             dttv2 = np.asarray([((2*DAT_Y[:,i+3])+(-27*DAT_Y[:,i+2]) + (270*DAT_Y[:,i+1]) - (490*DAT_Y[:,i]) + (270*DAT_Y[:,i-1]) - (27*DAT_Y[:,i-2]) + (2*DAT_Y[:,i-3]))/ (180*(dt**2)) for i in range(1,nt-1)])
#             dttv1 = np.asarray([((2*DAT_X[:,i+3])+(-27*DAT_X[:,i+2]) + (270*DAT_X[:,i+1]) - (490*DAT_X[:,i]) + (270*DAT_X[:,i-1]) - (27*DAT_X[:,i-2]) + (2*DAT_X[:,i-3]))/ (180*(dt**2)) for i in range(1,nt-1)])
       
    return dttv1, dttv2, dttv3


# -

def _1st_DER(FIELD_X, FIELD_Y, FIELD_Z, dx, dy, dz, nrx, nry, nrz,order):
    ## Initialize Gradients
    FIELD_X_gradX = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradX = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradX = np.zeros(FIELD_Z.shape)

    FIELD_X_gradY = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradY = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradY = np.zeros(FIELD_Z.shape)

    FIELD_X_gradZ = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradZ = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradZ = np.zeros(FIELD_Z.shape)
    
    FIELD_X_gradXY = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradXY = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradXY = np.zeros(FIELD_Z.shape)

    FIELD_X_gradYX = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradYX = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradYX = np.zeros(FIELD_Z.shape)

    FIELD_X_gradYZ = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradYZ = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradYZ = np.zeros(FIELD_Z.shape)

    FIELD_X_gradXZ = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradXZ = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradXZ = np.zeros(FIELD_Z.shape)
    
    if order == 2:
        for pos in range(1, nrx-1):
            FIELD_X_gradX[pos,:,:,:] = (FIELD_X[pos+1,:,:,:] - FIELD_X[pos-1,:,:,:])/(2*dx)
            FIELD_Y_gradX[pos,:,:,:] = (FIELD_Y[pos+1,:,:,:] - FIELD_Y[pos-1,:,:,:])/(2*dx)
            FIELD_Z_gradX[pos,:,:,:] = (FIELD_Z[pos+1,:,:,:] - FIELD_Z[pos-1,:,:,:])/(2*dx)

        for pos in range(1, nry-1):
            FIELD_X_gradY[:,pos,:,:] = (FIELD_X[:,pos+1,:,:] - FIELD_X[:,pos-1,:,:])/(2*dy)
            FIELD_Y_gradY[:,pos,:,:] = (FIELD_Y[:,pos+1,:,:] - FIELD_Y[:,pos-1,:,:])/(2*dy)
            FIELD_Z_gradY[:,pos,:,:] = (FIELD_Z[:,pos+1,:,:] - FIELD_Z[:,pos-1,:,:])/(2*dy)

        for pos in range(1, nrz-1):
            FIELD_X_gradZ[:,:,pos,:] = (-FIELD_X[:,:,pos+1,:] + FIELD_X[:,:,pos-1,:])/(2*dz)
            FIELD_Y_gradZ[:,:,pos,:] = (-FIELD_Y[:,:,pos+1,:] + FIELD_Y[:,:,pos-1,:])/(2*dz)
            FIELD_Z_gradZ[:,:,pos,:] = (-FIELD_Z[:,:,pos+1,:] + FIELD_Z[:,:,pos-1,:])/(2*dz)
                
            
    if order == 4:
        for pos in range(2, nrx-2):
            FIELD_X_gradX[pos,:,:,:] = (-FIELD_X[pos+2,:,:,:] + 8*FIELD_X[pos+1,:,:,:] - 8*FIELD_X[pos-1,:,:,:] + FIELD_X[pos-2,:,:,:])/(12*dx)
            FIELD_Y_gradX[pos,:,:,:] = (-FIELD_Y[pos+2,:,:,:] + 8*FIELD_Y[pos+1,:,:,:] - 8*FIELD_Y[pos-1,:,:,:] + FIELD_Y[pos-2,:,:,:])/(12*dx)
            FIELD_Z_gradX[pos,:,:,:] = (-FIELD_Z[pos+2,:,:,:] + 8*FIELD_Z[pos+1,:,:,:] - 8*FIELD_Z[pos-1,:,:,:] + FIELD_Z[pos-2,:,:,:])/(12*dx)

        for pos in range(2, nry-2):
            FIELD_X_gradY[:,pos,:,:] = (-FIELD_X[:,pos+2,:,:] + 8*FIELD_X[:,pos+1,:,:] - 8*FIELD_X[:,pos-1,:,:] + FIELD_X[:,pos-2,:,:])/(12*dy)
            FIELD_Y_gradY[:,pos,:,:] = (-FIELD_Y[:,pos+2,:,:] + 8*FIELD_Y[:,pos+1,:,:] - 8*FIELD_Y[:,pos-1,:,:] + FIELD_Y[:,pos-2,:,:])/(12*dy)
            FIELD_Z_gradY[:,pos,:,:] = (-FIELD_Z[:,pos+2,:,:] + 8*FIELD_Z[:,pos+1,:,:] - 8*FIELD_Z[:,pos-1,:,:] + FIELD_Z[:,pos-2,:,:])/(12*dy)

        for pos in range(2, nrz-2):
            FIELD_X_gradZ[:,:,pos,:] = (-FIELD_X[:,:,pos+2,:] + 8*FIELD_X[:,:,pos+1,:] - 8*FIELD_X[:,:,pos-1,:] + FIELD_X[:,:,pos-2,:])/(-12*dz)
            FIELD_Y_gradZ[:,:,pos,:] = (-FIELD_Y[:,:,pos+2,:] + 8*FIELD_Y[:,:,pos+1,:] - 8*FIELD_Y[:,:,pos-1,:] + FIELD_Y[:,:,pos-2,:])/(-12*dz)
            FIELD_Z_gradZ[:,:,pos,:] = (-FIELD_Z[:,:,pos+2,:] + 8*FIELD_Z[:,:,pos+1,:] - 8*FIELD_Z[:,:,pos-1,:] + FIELD_Z[:,:,pos-2,:])/(-12*dz)
        
        
    return FIELD_X_gradX, FIELD_Y_gradX, FIELD_Z_gradX, FIELD_X_gradY, FIELD_Y_gradY, FIELD_Z_gradY, \
           FIELD_X_gradZ, FIELD_Y_gradZ, FIELD_Z_gradZ


def ROT_2nd_DER(FIELD_Xx,FIELD_Xy,FIELD_Xz, FIELD_Yx,FIELD_Yy, FIELD_Yz, FIELD_Zx, FIELD_Zy, FIELD_Zz, dx, dy, dz, nrx, nry, nrz):
    ## Initialize Gradients
    FIELD_X_gradXX = np.zeros(FIELD_Zz.shape)
    FIELD_Y_gradXX = np.zeros(FIELD_Zz.shape)
    FIELD_Z_gradXX = np.zeros(FIELD_Zz.shape)

    FIELD_X_gradYY = np.zeros(FIELD_Zz.shape)
    FIELD_Y_gradYY = np.zeros(FIELD_Zz.shape)
    FIELD_Z_gradYY = np.zeros(FIELD_Zz.shape)

    FIELD_X_gradZZ = np.zeros(FIELD_Zz.shape)
    FIELD_Y_gradZZ = np.zeros(FIELD_Zz.shape)
    FIELD_Z_gradZZ = np.zeros(FIELD_Zz.shape)
    
    FIELD_X_gradXY = np.zeros(FIELD_Zz.shape)
    FIELD_Y_gradXY = np.zeros(FIELD_Zz.shape)
    FIELD_Z_gradXY = np.zeros(FIELD_Zz.shape)

    FIELD_X_gradYX = np.zeros(FIELD_Zz.shape)
    FIELD_Y_gradYX = np.zeros(FIELD_Zz.shape)
    FIELD_Z_gradYX = np.zeros(FIELD_Zz.shape)

    FIELD_X_gradYZ = np.zeros(FIELD_Zz.shape)
    FIELD_Y_gradYZ = np.zeros(FIELD_Zz.shape)
    FIELD_Z_gradYZ = np.zeros(FIELD_Zz.shape)

    FIELD_X_gradXZ = np.zeros(FIELD_Zz.shape)
    FIELD_Y_gradXZ = np.zeros(FIELD_Zz.shape)
    FIELD_Z_gradXZ = np.zeros(FIELD_Zz.shape)
    
    for pos in range(1, nrx-1):
        FIELD_X_gradXX[pos,:,:,:] = (FIELD_Xx[pos+1,:,:,:] - FIELD_Xx[pos-1,:,:,:])/(2*dx)
        FIELD_Y_gradXX[pos,:,:,:] = (FIELD_Yx[pos+1,:,:,:] - FIELD_Yx[pos-1,:,:,:])/(2*dx)
        FIELD_Z_gradXX[pos,:,:,:] = (FIELD_Zx[pos+1,:,:,:] - FIELD_Zx[pos-1,:,:,:])/(2*dx)
        
        FIELD_X_gradYX[pos,:,:,:] = (FIELD_Xy[pos+1,:,:,:] - FIELD_Xy[pos-1,:,:,:])/(2*dx)
        FIELD_Y_gradYX[pos,:,:,:] = (FIELD_Yy[pos+1,:,:,:] - FIELD_Yy[pos-1,:,:,:])/(2*dx)
        FIELD_Z_gradYX[pos,:,:,:] = (FIELD_Zy[pos+1,:,:,:] - FIELD_Zy[pos-1,:,:,:])/(2*dx)
        
    for pos in range(1, nry-1):
        FIELD_X_gradYY[:,pos,:,:] = (FIELD_Xy[:,pos+1,:,:] - FIELD_Xy[:,pos-1,:,:])/(2*dy)
        FIELD_Y_gradYY[:,pos,:,:] = (FIELD_Yy[:,pos+1,:,:] - FIELD_Yy[:,pos-1,:,:])/(2*dy)
        FIELD_Z_gradYY[:,pos,:,:] = (FIELD_Zy[:,pos+1,:,:] - FIELD_Zy[:,pos-1,:,:])/(2*dy)
        
        FIELD_X_gradXY[:,pos,:,:] = (FIELD_Xx[:,pos+1,:,:] - FIELD_Xx[:,pos-1,:,:])/(2*dy)
        FIELD_Y_gradXY[:,pos,:,:] = (FIELD_Yx[:,pos+1,:,:] - FIELD_Yx[:,pos-1,:,:])/(2*dy)
        FIELD_Z_gradXY[:,pos,:,:] = (FIELD_Zx[:,pos+1,:,:] - FIELD_Zx[:,pos-1,:,:])/(2*dy)
        
    for pos in range(1, nrz-1):
        FIELD_X_gradZZ[:,:,pos,:] = (-FIELD_Xz[:,:,pos+1,:] + FIELD_Xz[:,:,pos-1,:])/(2*dz)
        FIELD_Y_gradZZ[:,:,pos,:] = (-FIELD_Yz[:,:,pos+1,:] + FIELD_Yz[:,:,pos-1,:])/(2*dz)
        FIELD_Z_gradZZ[:,:,pos,:] = (-FIELD_Zz[:,:,pos+1,:] + FIELD_Zz[:,:,pos-1,:])/(2*dz)
        
        FIELD_X_gradYZ[:,:,pos,:] = (-FIELD_Xy[:,:,pos+1,:] + FIELD_Xy[:,:,pos-1,:])/(2*dz)
        FIELD_Y_gradYZ[:,:,pos,:] = (-FIELD_Yy[:,:,pos+1,:] + FIELD_Yy[:,:,pos-1,:])/(2*dz)
        FIELD_Z_gradYZ[:,:,pos,:] = (-FIELD_Zy[:,:,pos+1,:] + FIELD_Zy[:,:,pos-1,:])/(2*dz)
        
        FIELD_X_gradXZ[:,:,pos,:] = (-FIELD_Xx[:,:,pos+1,:] + FIELD_Xx[:,:,pos-1,:])/(2*dz)
        FIELD_Y_gradXZ[:,:,pos,:] = (-FIELD_Yx[:,:,pos+1,:] + FIELD_Yx[:,:,pos-1,:])/(2*dz)
        FIELD_Z_gradXZ[:,:,pos,:] = (-FIELD_Zx[:,:,pos+1,:] + FIELD_Zx[:,:,pos-1,:])/(2*dz)
        
        
    return FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
           FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
           FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ


def _2nd_DER(FIELD_X, FIELD_Y, FIELD_Z, dx, dy, dz, nrx, nry, nrz, order):
    ## Initialize Gradients
   
    FIELD_X_gradXX = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradXX = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradXX = np.zeros(FIELD_Z.shape)

    FIELD_X_gradYY = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradYY = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradYY = np.zeros(FIELD_Z.shape)

    FIELD_X_gradZZ = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradZZ = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradZZ = np.zeros(FIELD_Z.shape)

    FIELD_X_gradXY = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradXY = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradXY = np.zeros(FIELD_Z.shape)

    FIELD_X_gradYX = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradYX = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradYX = np.zeros(FIELD_Z.shape)

    FIELD_X_gradYZ = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradYZ = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradYZ = np.zeros(FIELD_Z.shape)

    FIELD_X_gradXZ = np.zeros(FIELD_Z.shape)
    FIELD_Y_gradXZ = np.zeros(FIELD_Z.shape)
    FIELD_Z_gradXZ = np.zeros(FIELD_Z.shape)
    
    # 2nd order accurate
    if order == 2:
        for pos in range(1, nrx-1):
                FIELD_X_gradXX[pos,:,:,:] = (FIELD_X[pos+1,:,:,:] - (2* FIELD_X[pos,:,:,:]) + FIELD_X[pos-1,:,:,:])/(dx**2)
                FIELD_Y_gradXX[pos,:,:,:] = (FIELD_Y[pos+1,:,:,:] - (2* FIELD_Y[pos,:,:,:]) + FIELD_Y[pos-1,:,:,:])/(dx**2)
                FIELD_Z_gradXX[pos,:,:,:] = (FIELD_Z[pos+1,:,:,:] - (2* FIELD_Z[pos,:,:,:]) + FIELD_Z[pos-1,:,:,:])/(dx**2)

        for pos in range(1, nry-1):
                FIELD_X_gradYY[:,pos,:,:] = (FIELD_X[:,pos+1,:,:] -  (2* FIELD_X[:,pos,:,:]) +  FIELD_X[:,pos-1,:,:])/(dy**2)
                FIELD_Y_gradYY[:,pos,:,:] = (FIELD_Y[:,pos+1,:,:] -  (2* FIELD_Y[:,pos,:,:]) +  FIELD_Y[:,pos-1,:,:])/(dy**2)
                FIELD_Z_gradYY[:,pos,:,:] = (FIELD_Z[:,pos+1,:,:] -  (2* FIELD_Z[:,pos,:,:]) +  FIELD_Z[:,pos-1,:,:])/(dy**2)

        for pos in range(1, nrz-1):
                FIELD_X_gradZZ[:,:,pos,:] = (FIELD_X[:,:,pos+1,:] - (2* FIELD_X[:,:,pos,:]) + FIELD_X[:,:,pos-1,:])/(dz**2)
                FIELD_Y_gradZZ[:,:,pos,:] = (FIELD_Y[:,:,pos+1,:] - (2* FIELD_Y[:,:,pos,:]) + FIELD_Y[:,:,pos-1,:])/(dz**2)
                FIELD_Z_gradZZ[:,:,pos,:] = (FIELD_Z[:,:,pos+1,:] - (2* FIELD_Z[:,:,pos,:]) + FIELD_Z[:,:,pos-1,:])/(dz**2)


        for posx in range(1, nrx-1):
            for posy in range(1, nry-1):
                FIELD_X_gradXY[posx,posy,:,:] = (FIELD_X[posx+1,posy+1,:,:] - FIELD_X[posx-1,posy+1,:,:] - \
                                            FIELD_X[posx+1,posy-1,:,:] + FIELD_X[posx-1,posy-1,:,:])/(4*dx*dy)
                FIELD_Y_gradXY[posx,posy,:,:] = (FIELD_Y[posx+1,posy+1,:,:] - FIELD_Y[posx-1,posy+1,:,:] - \
                                            FIELD_Y[posx+1,posy-1,:,:] + FIELD_Y[posx-1,posy-1,:,:])/(4*dx*dy)
                FIELD_Z_gradXY[posx,posy,:,:] = (FIELD_Z[posx+1,posy+1,:,:] - FIELD_Z[posx-1,posy+1,:,:] - \
                                            FIELD_Z[posx+1,posy-1,:,:] + FIELD_Z[posx-1,posy-1,:,:])/(4*dx*dy)

        FIELD_X_gradYX = FIELD_X_gradXY
        FIELD_Y_gradYX = FIELD_Y_gradXY
        FIELD_Z_gradYX = FIELD_Z_gradXY



        for posz in range(1, nrz-1):
            for posy in range(1, nry-1):
                FIELD_X_gradYZ[:,posy,posz,:] = (FIELD_X[:,posy+1,posz+1,:] - FIELD_X[:,posy+1,posz-1,:] - \
                                                    FIELD_X[:,posy-1,posz+1,:] + FIELD_X[:,posy-1,posz-1,:])/-(4*dz*dy)
                FIELD_Y_gradYZ[:,posy,posz,:] = (FIELD_Y[:,posy+1,posz+1,:] - FIELD_Y[:,posy+1,posz-1,:] - \
                                                    FIELD_Y[:,posy-1,posz+1,:] + FIELD_Y[:,posy-1,posz-1,:])/-(4*dz*dy)
                FIELD_Z_gradYZ[:,posy,posz,:] = (FIELD_Z[:,posy+1,posz+1,:] - FIELD_Z[:,posy+1,posz-1,:] - \
                                                    FIELD_Z[:,posy-1,posz+1,:] + FIELD_Z[:,posy-1,posz-1,:])/-(4*dz*dy)


        for posz in range(1, nrz-1):
            for posx in range(1, nrx-1):
                FIELD_X_gradXZ[posx,:,posz,:] = (FIELD_X[posx+1,:,posz+1,:] - FIELD_X[posx+1,:,posz-1,:] - \
                                                    FIELD_X[posx-1,:,posz+1,:] + FIELD_X[posx-1,:,posz-1,:])/-(4*dz*dx)
                FIELD_Y_gradXZ[posx,:,posz,:] = (FIELD_Y[posx+1,:,posz+1,:] - FIELD_Y[posx+1,:,posz-1,:] - \
                                                    FIELD_Y[posx-1,:,posz+1,:] + FIELD_Y[posx-1,:,posz-1,:])/-(4*dz*dx)
                FIELD_Z_gradXZ[posx,:,posz,:] = (FIELD_Z[posx+1,:,posz+1,:] - FIELD_Z[posx+1,:,posz-1,:] - \
                                                    FIELD_Z[posx-1,:,posz+1,:] + FIELD_Z[posx-1,:,posz-1,:])/-(4*dz*dx)

    if order == 4:
        for pos in range(2, nrx-2):
                FIELD_X_gradXX[pos,:,:,:] = (-FIELD_X[pos+2,:,:,:] + 16*FIELD_X[pos+1,:,:,:] - 30*FIELD_X[pos,:,:,:] \
                                             + 16*FIELD_X[pos-1,:,:,:] -  FIELD_X[pos-2,:,:,:] )/(12*(dx**2))
                FIELD_Y_gradXX[pos,:,:,:] = (-FIELD_Y[pos+2,:,:,:] + 16*FIELD_Y[pos+1,:,:,:] - 30*FIELD_Y[pos,:,:,:] \
                                             + 16*FIELD_Y[pos-1,:,:,:] -  FIELD_Y[pos-2,:,:,:] )/(12*(dx**2))
                FIELD_Z_gradXX[pos,:,:,:] = (-FIELD_Z[pos+2,:,:,:] + 16*FIELD_Z[pos+1,:,:,:] - 30*FIELD_Z[pos,:,:,:] \
                                             + 16*FIELD_Z[pos-1,:,:,:] -  FIELD_Z[pos-2,:,:,:] )/(12*(dx**2))

        for pos in range(2, nry-2):
                FIELD_X_gradYY[:,pos,:,:] = (-FIELD_X[:,pos+2,:,:] + 16*FIELD_X[:,pos+1,:,:] - 30*FIELD_X[:,pos,:,:] \
                                             + 16*FIELD_X[:,pos-1,:,:] -  FIELD_X[:,pos-2,:,:] )/(12*(dy**2))
                FIELD_Y_gradYY[:,pos,:,:] = (-FIELD_Y[:,pos+2,:,:] + 16*FIELD_Y[:,pos+1,:,:] - 30*FIELD_Y[:,pos,:,:] \
                                             + 16*FIELD_Y[:,pos-1,:,:] -  FIELD_Y[:,pos-2,:,:] )/(12*(dy**2))
                FIELD_Z_gradYY[:,pos,:,:] = (-FIELD_Z[:,pos+2,:,:] + 16*FIELD_Z[:,pos+1,:,:] - 30*FIELD_Z[:,pos,:,:] \
                                             + 16*FIELD_Z[:,pos-1,:,:] -  FIELD_Z[:,pos-2,:,:] )/(12*(dy**2))


        for pos in range(2, nrz-2):
                FIELD_X_gradZZ[:,:,pos,:] = (-FIELD_X[:,:,pos+2,:] + 16*FIELD_X[:,:,pos+1,:] - 30*FIELD_X[:,:,pos,:] \
                                             + 16*FIELD_X[:,:,pos-1,:] -  FIELD_X[:,:,pos-2,:] )/(12*(dz**2))
                FIELD_Y_gradZZ[:,:,pos,:] = (-FIELD_Y[:,:,pos+2,:] + 16*FIELD_Y[:,:,pos+1,:] - 30*FIELD_Y[:,:,pos,:] \
                                             + 16*FIELD_Y[:,:,pos-1,:] -  FIELD_Y[:,:,pos-2,:] )/(12*(dz**2))
                FIELD_Z_gradZZ[:,:,pos,:] = (-FIELD_Z[:,:,pos+2,:] + 16*FIELD_Z[:,:,pos+1,:] - 30*FIELD_Z[:,:,pos,:] \
                                             + 16*FIELD_Z[:,:,pos-1,:] -  FIELD_Z[:,:,pos-2,:] )/(12*(dz**2))
                



        for posx in range(2, nrx-2):
            for posy in range(2, nry-2):
                FIELD_X_gradXY[posx,posy,:,:] = (8*(FIELD_X[posx+1,posy-2,:,:]+FIELD_X[posx+2,posy-1,:,:]+FIELD_X[posx-2,posy+1,:,:]+FIELD_X[posx-1,posy+2,:,:])\
                                                - 8*(FIELD_X[posx-1,posy-2,:,:]+FIELD_X[posx-2,posy-1,:,:]+FIELD_X[posx+1,posy+2,:,:]+FIELD_X[posx+2,posy+1,:,:])\
                                                - (FIELD_X[posx+2,posy-2,:,:]+FIELD_X[posx-2,posy+2,:,:]-FIELD_X[posx-2,posy-2,:,:]-FIELD_X[posx+2,posy+2,:,:])\
                                                + 64 * (FIELD_X[posx-1,posy-1,:,:]+FIELD_X[posx+1,posy+1,:,:]-FIELD_X[posx+1,posy-1,:,:]-FIELD_X[posx-1,posy+1,:,:]))/(144*dx*dy)
                
                FIELD_Y_gradXY[posx,posy,:,:] = (8*(FIELD_Y[posx+1,posy-2,:,:]+FIELD_Y[posx+2,posy-1,:,:]+FIELD_Y[posx-2,posy+1,:,:]+FIELD_Y[posx-1,posy+2,:,:])\
                                                - 8*(FIELD_Y[posx-1,posy-2,:,:]+FIELD_Y[posx-2,posy-1,:,:]+FIELD_Y[posx+1,posy+2,:,:]+FIELD_Y[posx+2,posy+1,:,:])\
                                                - (FIELD_Y[posx+2,posy-2,:,:]+FIELD_Y[posx-2,posy+2,:,:]-FIELD_Y[posx-2,posy-2,:,:]-FIELD_Y[posx+2,posy+2,:,:])\
                                                + 64 * (FIELD_Y[posx-1,posy-1,:,:]+FIELD_Y[posx+1,posy+1,:,:]-FIELD_Y[posx+1,posy-1,:,:]-FIELD_Y[posx-1,posy+1,:,:]))/(144*dx*dy)
                
                FIELD_Z_gradXY[posx,posy,:,:] = (8*(FIELD_Z[posx+1,posy-2,:,:]+FIELD_Z[posx+2,posy-1,:,:]+FIELD_Z[posx-2,posy+1,:,:]+FIELD_Z[posx-1,posy+2,:,:])\
                                                - 8*(FIELD_Z[posx-1,posy-2,:,:]+FIELD_Z[posx-2,posy-1,:,:]+FIELD_Z[posx+1,posy+2,:,:]+FIELD_Z[posx+2,posy+1,:,:])\
                                                - (FIELD_Z[posx+2,posy-2,:,:]+FIELD_Z[posx-2,posy+2,:,:]-FIELD_Z[posx-2,posy-2,:,:]-FIELD_Z[posx+2,posy+2,:,:])\
                                                + 64 * (FIELD_Z[posx-1,posy-1,:,:]+FIELD_Z[posx+1,posy+1,:,:]-FIELD_Z[posx+1,posy-1,:,:]-FIELD_Z[posx-1,posy+1,:,:]))/(144*dx*dy)

        FIELD_X_gradYX = FIELD_X_gradXY
        FIELD_Y_gradYX = FIELD_Y_gradXY
        FIELD_Z_gradYX = FIELD_Z_gradXY



        for posz in range(2, nrz-2):
            for posy in range(2, nry-2):
                FIELD_X_gradYZ[:,posy,posz,:] = (8*(FIELD_X[:,posy+1,posz-2,:]+FIELD_X[:,posy+2,posz-1,:]+FIELD_X[:,posy-2,posz+1,:]+FIELD_X[:,posy-1,posz+2,:])\
                                                - 8*(FIELD_X[:,posy-1,posz-2,:]+FIELD_X[:,posy-2,posz-1,:]+FIELD_X[:,posy+1,posz+2,:]+FIELD_X[:,posy+2,posz+1,:])\
                                                - (FIELD_X[:,posy+2,posz-2,:]+FIELD_X[:,posy-2,posz+2,:]-FIELD_X[:,posy-2,posz-2,:]-FIELD_X[:,posy+2,posz+2,:])\
                                                + 64 * (FIELD_X[:,posy-1,posz-1,:]+FIELD_X[:,posy+1,posz+1,:]-FIELD_X[:,posy+1,posz-1,:]-FIELD_X[:,posy-1,posz+1,:]))/-(144*dz*dy)
                
                FIELD_Y_gradYZ[:,posy,posz,:] = (8*(FIELD_Y[:,posy+1,posz-2,:]+FIELD_Y[:,posy+2,posz-1,:]+FIELD_Y[:,posy-2,posz+1,:]+FIELD_Y[:,posy-1,posz+2,:])\
                                                - 8*(FIELD_Y[:,posy-1,posz-2,:]+FIELD_Y[:,posy-2,posz-1,:]+FIELD_Y[:,posy+1,posz+2,:]+FIELD_Y[:,posy+2,posz+1,:])\
                                                - (FIELD_Y[:,posy+2,posz-2,:]+FIELD_Y[:,posy-2,posz+2,:]-FIELD_Y[:,posy-2,posz-2,:]-FIELD_Y[:,posy+2,posz+2,:])\
                                                + 64 * (FIELD_Y[:,posy-1,posz-1,:]+FIELD_Y[:,posy+1,posz+1,:]-FIELD_Y[:,posy+1,posz-1,:]-FIELD_Y[:,posy-1,posz+1,:]))/-(144*dz*dy)
                
                FIELD_Z_gradYZ[:,posy,posz,:] = (8*(FIELD_Z[:,posy+1,posz-2,:]+FIELD_Z[:,posy+2,posz-1,:]+FIELD_Z[:,posy-2,posz+1,:]+FIELD_Z[:,posy-1,posz+2,:])\
                                                - 8*(FIELD_Z[:,posy-1,posz-2,:]+FIELD_Z[:,posy-2,posz-1,:]+FIELD_Z[:,posy+1,posz+2,:]+FIELD_Z[:,posy+2,posz+1,:])\
                                                - (FIELD_Z[:,posy+2,posz-2,:]+FIELD_Z[:,posy-2,posz+2,:]-FIELD_Z[:,posy-2,posz-2,:]-FIELD_Z[:,posy+2,posz+2,:])\
                                                + 64 * (FIELD_Z[:,posy-1,posz-1,:]+FIELD_Z[:,posy+1,posz+1,:]-FIELD_Z[:,posy+1,posz-1,:]-FIELD_Z[:,posy-1,posz+1,:]))/-(144*dz*dy)


        for posz in range(2, nrz-2):
            for posx in range(2, nrx-2):
                FIELD_X_gradXZ[posx,:,posz,:] = (8*(FIELD_X[posx+1,:,posz-2,:]+FIELD_X[posx+2,:,posz-1,:]+FIELD_X[posx-2,:,posz+1,:]+FIELD_X[posx-1,:,posz+2,:])\
                                                - 8*(FIELD_X[posx-1,:,posz-2,:]+FIELD_X[posx-2,:,posz-1,:]+FIELD_X[posx+1,:,posz+2,:]+FIELD_X[posx+2,:,posz+1,:])\
                                                - (FIELD_X[posx+2,:,posz-2,:]+FIELD_X[posx-2,:,posz+2,:]-FIELD_X[posx-2,:,posz-2,:]-FIELD_X[posx+2,:,posz+2,:])\
                                                + 64 * (FIELD_X[posx-1,:,posz-1,:]+FIELD_X[posx+1,:,posz+1,:]-FIELD_X[posx+1,:,posz-1,:]-FIELD_X[posx-1,:,posz+1,:]))/-(144*dz*dx)
                
                FIELD_Y_gradXZ[posx,:,posz,:] = (8*(FIELD_Y[posx+1,:,posz-2,:]+FIELD_Y[posx+2,:,posz-1,:]+FIELD_Y[posx-2,:,posz+1,:]+FIELD_Y[posx-1,:,posz+2,:])\
                                                - 8*(FIELD_Y[posx-1,:,posz-2,:]+FIELD_Y[posx-2,:,posz-1,:]+FIELD_Y[posx+1,:,posz+2,:]+FIELD_Y[posx+2,:,posz+1,:])\
                                                - (FIELD_Y[posx+2,:,posz-2,:]+FIELD_Y[posx-2,:,posz+2,:]-FIELD_Y[posx-2,:,posz-2,:]-FIELD_Y[posx+2,:,posz+2,:])\
                                                + 64 * (FIELD_Y[posx-1,:,posz-1,:]+FIELD_Y[posx+1,:,posz+1,:]-FIELD_Y[posx+1,:,posz-1,:]-FIELD_Y[posx-1,:,posz+1,:]))/-(144*dz*dx)
                
                FIELD_Z_gradXZ[posx,:,posz,:] = (8*(FIELD_Z[posx+1,:,posz-2,:]+FIELD_Z[posx+2,:,posz-1,:]+FIELD_Z[posx-2,:,posz+1,:]+FIELD_Z[posx-1,:,posz+2,:])\
                                                - 8*(FIELD_Z[posx-1,:,posz-2,:]+FIELD_Z[posx-2,:,posz-1,:]+FIELD_Z[posx+1,:,posz+2,:]+FIELD_Z[posx+2,:,posz+1,:])\
                                                - (FIELD_Z[posx+2,:,posz-2,:]+FIELD_Z[posx-2,:,posz+2,:]-FIELD_Z[posx-2,:,posz-2,:]-FIELD_Z[posx+2,:,posz+2,:])\
                                                + 64 * (FIELD_Z[posx-1,:,posz-1,:]+FIELD_Z[posx+1,:,posz+1,:]-FIELD_Z[posx+1,:,posz-1,:]-FIELD_Z[posx-1,:,posz+1,:]))/-(144*dz*dx)

        
    return FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
           FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
           FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ



