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
import matplotlib.pyplot as plt


def LINeq_VAR(dz, FIELD_X_gradX, FIELD_Y_gradY, FIELD_Z_gradZ_FDnotFS, FIELD_Z_gradXX, FIELD_Z_gradYY, FIELD_Z_gradY, \
              FIELD_Y_gradZ_FDnotFS, FIELD_Y_gradXX, FIELD_Y_gradYY, FIELD_X_gradYX, FIELD_Z_gradX, \
              FIELD_X_gradZ_FDnotFS, FIELD_X_gradXX, FIELD_X_gradYY, FIELD_Y_gradXY):
    
    A3 = (2/dz * ((FIELD_X_gradX+FIELD_Y_gradY) + FIELD_Z_gradZ_FDnotFS) ) - (FIELD_Z_gradXX + FIELD_Z_gradYY)
    B3 = (4/dz *  (FIELD_X_gradX+FIELD_Y_gradY) ) - (2*(FIELD_Z_gradXX + FIELD_Z_gradYY))
    
    A2 = (2/dz * (FIELD_Z_gradY + FIELD_Y_gradZ_FDnotFS)) + (FIELD_Y_gradXX + FIELD_Y_gradYY) + (2*(FIELD_X_gradYX + FIELD_Y_gradYY))
    B2 = 2*(FIELD_X_gradYX + FIELD_Y_gradYY)
    
    A1 = (2/dz * (FIELD_Z_gradX + FIELD_X_gradZ_FDnotFS)) + ( FIELD_X_gradXX + FIELD_X_gradYY) + (2*( FIELD_X_gradXX + FIELD_Y_gradXY))
    B1 = 2*(FIELD_X_gradXX + FIELD_Y_gradXY)
    
    return A1, B1, A2, B2, A3, B3


def WE_3D_ELASTIC_FreeSurf(A1, A2, A3, B1, B2, B3, vp_sub, vs_sub, dttv1, dttv2, dttv3, posX,posY,t_range):
    
    ## Check if they fit for true velocity model
    
    rhs_eq1 = ((vs_sub[:,:, np.newaxis]**2) * A1) - (((vs_sub[:,:, np.newaxis]**4)/(vp_sub[:,:, np.newaxis]**2)) *B1)
    rhs_eq2 = ((vs_sub[:,:, np.newaxis]**2) * A2) - (((vs_sub[:,:, np.newaxis]**4)/(vp_sub[:,:, np.newaxis]**2)) *B2)
    rhs_eq3 = ((vp_sub[:,:, np.newaxis]**2) * A3) - ((vs_sub[:,:, np.newaxis]**2) *B3)
    
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,10),nrows=3)
    #fig.subplots_adjust(wspace=50)
    plt.suptitle('Test 3D elastic Free Surface Correction')
    ax1.set_title('Difference at the order of '+str(abs(rhs_eq1[posX,posY,:]-dttv1[posX,posY,:]).sum()))
    ax1.plot(t_range,rhs_eq1[posX,posY,:], label='RHS Free Surface with FD gradients')
    ax1.plot(t_range,dttv1[posX,posY,:], label='Temporal derivative of displacement', linestyle='-.')
    ax1.set_ylabel('X - component')
    ax1.legend()

    ax2.plot(t_range,rhs_eq2[posX,posY,:])
    ax2.plot(t_range,dttv2[posX,posY,:], linestyle='-.')
    ax2.set_ylabel('Y - component')

    ax3.plot(t_range,rhs_eq3[posX,posY,:])
    ax3.plot(t_range,dttv3[posX,posY,:], linestyle='-.')
    ax3.set_ylabel('Z - component')
    return rhs_eq3, rhs_eq2, rhs_eq1


def WE_3D_ELASTIC_FDnotFS(FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
                FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
                FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ,\
                vp_sub, vs_sub, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
    
    rotROT1 = FIELD_Y_gradXY - FIELD_X_gradYY - FIELD_X_gradZZ + FIELD_Z_gradXZ
    rotROT2 = -  FIELD_Y_gradZZ + FIELD_Z_gradYZ + FIELD_X_gradXY - FIELD_Y_gradXX
    rotROT3 = FIELD_Y_gradYZ - FIELD_Z_gradYY + FIELD_X_gradXZ - FIELD_Z_gradXX

    gradDIV1 = FIELD_X_gradXX + FIELD_Y_gradXY + FIELD_Z_gradXZ
    gradDIV2 = FIELD_X_gradXY + FIELD_Y_gradYY + FIELD_Z_gradYZ
    gradDIV3 = FIELD_X_gradXZ + FIELD_Y_gradYZ + FIELD_Z_gradZZ

    
    ## Check if they fit for true velocity model
    
    rhs_eq1 = ((vp_sub[:,:,  np.newaxis]**2) * gradDIV1) - ((vs_sub[:,:,  np.newaxis]**2) *rotROT1)
    rhs_eq2 = ((vp_sub[:,:,  np.newaxis]**2) * gradDIV2) - ((vs_sub[:,:,  np.newaxis]**2) *rotROT2)
    rhs_eq3 = ((vp_sub[:,:,  np.newaxis]**2) * gradDIV3) - ((vs_sub[:,:,  np.newaxis]**2) *rotROT3)
    
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,10),nrows=3)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Test 3D elastic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Test 3D elastic Wave Equation (unfiltered)')
    #ax1.set_title('Difference at the order of '+str(abs(rhs_eq1[posX,posY,:,nf]-dttv1[posX,posY,:,nf]).sum()))
    ax1.plot(rhs_eq1[posX,posY,:], label='RHS elastic WE with FD gradients')
    ax1.plot(dttv1[posX,posY,:], label='Temporal derivative of displacement', linestyle='-.')
    ax1.set_ylabel('X - component')
    ax1.legend()

    ax2.plot(rhs_eq2[posX,posY,:])
    ax2.plot(dttv2[posX,posY,:], linestyle='-.')
    ax2.set_ylabel('Y - component')

    ax3.plot(rhs_eq3[posX,posY,:])
    ax3.plot(dttv3[posX,posY,:], linestyle='-.')
    ax3.set_ylabel('Z - component')

    return rotROT1, rotROT2, rotROT3, gradDIV1, gradDIV2, gradDIV3, rhs_eq1
