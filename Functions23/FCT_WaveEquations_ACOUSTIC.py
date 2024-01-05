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


def WE_3D_ACOUSTIC(FIELD_Z_gradXX, FIELD_Z_gradYY, FIELD_Z_gradZZ, FIELD_Z_gradXY, FIELD_Z_gradYX,  \
                   FIELD_Z_gradYZ, FIELD_Z_gradXZ, vp_sub, c_sub, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
  
    acc3 = FIELD_Z_gradXX + FIELD_Z_gradYY + FIELD_Z_gradZZ 


    ## Check if they fit for true velocity model
    
    rhs_eq3 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) *acc3)
    
    fig, (ax3) = plt.subplots(figsize=(12,8),nrows=1)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.title('Test 3D acoustic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.title('Test 3D acoustic Wave Equation (unfiltered)')

    ax3.plot(rhs_eq3[posX,posY,posZ,:,nf], label='lhs')
    ax3.plot(dttv3[posX,posY,posZ,:,nf], linestyle='-.', label ='dttv')
    ax3.set_ylabel('Z - component')
    plt.legend()
    return acc3


def WE_2D_ACOUSTIC_withRHO(FIELD_Z_gradXXYY, \
                           vp_sub, c_sub, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
  
    acc3_withRHO = FIELD_Z_gradXXYY


    ## Check if they fit for true velocity model
    
    rhs_eq3 = ((vp_sub[:,:,np.newaxis, np.newaxis]**2) *acc3_withRHO)
    
    fig, (ax3) = plt.subplots(figsize=(12,8),nrows=1)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.title('Test 2D FULL acoustic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.title('Test 2D FULL acoustic Wave Equation (unfiltered)')

    ax3.plot(rhs_eq3[posX,posY,:,nf], label ='lhs')
    ax3.plot(dttv3[posX,posY,posZ,:,nf], linestyle='-.', label ='dttv')
    ax3.set_ylabel('Z - component')
    plt.legend()
    return acc3_withRHO


def WE_2D_ACOUSTIC(FIELD_Z_gradXX, FIELD_Z_gradYY, FIELD_Z_gradZZ, FIELD_Z_gradXY, FIELD_Z_gradYX,  \
                   FIELD_Z_gradYZ, FIELD_Z_gradXZ, vp_sub, c_sub, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
  
    acc3_2D = FIELD_Z_gradXX + FIELD_Z_gradYY 


    ## Check if they fit for true velocity model
    
    rhs_eq3_2D = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) *acc3_2D)
    
    fig, (ax3) = plt.subplots(figsize=(12,8),nrows=1)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.title('Test 2D acoustic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.title('Test 2D acoustic Wave Equation (unfiltered)')

    ax3.plot(rhs_eq3_2D[posX,posY,posZ,:,nf], label='lhs')
    ax3.plot(dttv3[posX,posY,posZ,:,nf], linestyle='-.', label ='dttv')
    ax3.set_ylabel('Z - component')
    plt.legend()
    return acc3_2D
