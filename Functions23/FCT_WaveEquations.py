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


def WE_3D_ELASTIC(FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
                FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
                FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ,\
                vp_sub, vs_sub, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered,t_range):
    
    rotROT1 = FIELD_Y_gradXY - FIELD_X_gradYY - FIELD_X_gradZZ + FIELD_Z_gradXZ
    rotROT2 = -  FIELD_Y_gradZZ + FIELD_Z_gradYZ + FIELD_X_gradXY - FIELD_Y_gradXX
    rotROT3 = FIELD_Y_gradYZ - FIELD_Z_gradYY + FIELD_X_gradXZ - FIELD_Z_gradXX

    gradDIV1 = FIELD_X_gradXX + FIELD_Y_gradXY + FIELD_Z_gradXZ
    gradDIV2 = FIELD_X_gradXY + FIELD_Y_gradYY + FIELD_Z_gradYZ
    gradDIV3 = FIELD_X_gradXZ + FIELD_Y_gradYZ + FIELD_Z_gradZZ

    
    ## Check if they fit for true velocity model
    
    rhs_eq1 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV1) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT1)
    rhs_eq2 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV2) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT2)
    rhs_eq3 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV3) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT3)
    
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(14,12),nrows=3)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Test 3D elastic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Test 3D elastic Wave Equation (unfiltered)')
    ax1.set_title('Difference at the order of '+str(abs(rhs_eq1[posX,posY,posZ,:]-dttv1[posX,posY,posZ,:]).sum()))
    ax1.plot(t_range,rhs_eq1[posX,posY,posZ,:,nf], label=r'$LHS_{true}$')# elastic WE with FD gradients')
    ax1.plot(t_range,dttv1[posX,posY,posZ,:,nf], label=r'$RHS_{true}$', linestyle='-.')#Temporal derivative of displacement', linestyle='-.')
    ax1.set_ylabel('X - component')
    ax1.legend()

    ax2.plot(t_range,rhs_eq2[posX,posY,posZ,:,nf])
    ax2.plot(t_range,dttv2[posX,posY,posZ,:,nf], linestyle='-.')
    ax2.set_ylabel('Y - component')

    ax3.plot(t_range,rhs_eq3[posX,posY,posZ,:,nf])
    ax3.plot(t_range,dttv3[posX,posY,posZ,:,nf], linestyle='-.')
    ax3.set_ylabel('Z - component')
    ax3.set_xlabel('Time [s]')

    return rotROT1, rotROT2, rotROT3, gradDIV1, gradDIV2, gradDIV3


def WE_2D_ELASTIC_OPTi_SH(FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
                FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
                FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ,\
                vp_sub, vs_sub, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
    
    # Substitute gradients in 2D Elastic wave equation !!
    rotROT2_2D = FIELD_Y_gradXX + FIELD_Y_gradZZ
    rotROT1_2D = np.zeros(rotROT2_2D.shape)
    rotROT3_2D = np.zeros(rotROT2_2D.shape)
    
    gradDIV1_2D = np.zeros(rotROT2_2D.shape)
    gradDIV2_2D = np.zeros(rotROT2_2D.shape)
    gradDIV3_2D = np.zeros(rotROT2_2D.shape)

    
    rhs_eq2_2D =  ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT2_2D)
    

    fig, (ax1) = plt.subplots(figsize=(12,10),nrows=1)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Test 2D elastic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Test 2D elastic Wave Equation (unfiltered)')
    ax1.set_title('Difference at the order of '+str(abs(rhs_eq2_2D[posX,posY,posZ,:]-dttv2[posX,posY,posZ,:]).sum()))    
    ax1.plot(rhs_eq2_2D[posX,posY,posZ,:,nf], label='RHS elastic WE with FD gradients')
    ax1.plot(dttv2[posX,posY,posZ,:,nf], label='Temporal derivative of FIELDlacement', linestyle='-.')
    ax1.set_ylabel('Y - component')
    ax1.legend()

    

    
    return rotROT1_2D, rotROT2_2D, rotROT3_2D, gradDIV1_2D, gradDIV2_2D, gradDIV3_2D


def WE_2D_ELASTIC_OPTii_PSV(FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
                FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
                FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ,\
                vp_sub, vs_sub, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
    
    # Substitute gradients in 2D Elastic wave equation !!

    gradDIV1_2D = FIELD_X_gradXX - FIELD_Z_gradXZ
    gradDIV2_2D = np.zeros(gradDIV1_2D.shape)
    gradDIV3_2D = FIELD_Z_gradZZ - FIELD_X_gradXZ

    rotROT1_2D = FIELD_X_gradZZ - FIELD_Z_gradXZ
    rotROT2_2D = np.zeros(gradDIV1_2D.shape)
    rotROT3_2D = FIELD_Z_gradXX - FIELD_X_gradXZ


    rhs_eq1_2D = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV1_2D) + ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT1_2D)
    rhs_eq3_2D = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV3_2D) + ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT3_2D)
    

    fig, (ax1,ax2) = plt.subplots(figsize=(12,10),nrows=2)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Test 2D elastic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Test 2D elastic Wave Equation (unfiltered)')
    ax1.set_title('Difference at the order of '+str(abs(rhs_eq1_2D[posX,posY,posZ,:]-dttv1[posX,posY,posZ,:]).sum()))    
    ax1.plot(rhs_eq1_2D[posX,posY,posZ,:,nf], label='RHS elastic WE with FD gradients')
    ax1.plot(dttv1[posX,posY,posZ,:,nf], label='Temporal derivative of FIELDlacement', linestyle='-.')
    ax1.set_ylabel('X - component')
    ax1.legend()

    ax2.plot(rhs_eq3_2D[posX,posY,posZ,:,nf])
    ax2.plot(dttv3[posX,posY,posZ,:,nf], linestyle='-.')
    ax2.set_ylabel('Z - component')

    
    return rotROT1_2D, rotROT2_2D, rotROT3_2D, gradDIV1_2D, gradDIV2_2D, gradDIV3_2D


def WE_2D_ELASTIC_OPTiii_no_dz(FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
                FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
                FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ,\
                vp_sub, vs_sub, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
    
    # Substitute gradients in 2D Elastic wave equation !!

    rotROT1_2D = FIELD_Y_gradXY - FIELD_X_gradYY
    rotROT2_2D = FIELD_X_gradXY - FIELD_Y_gradXX
    rotROT3_2D = FIELD_Z_gradYY + FIELD_Z_gradXX

    gradDIV1_2D = FIELD_X_gradXX + FIELD_Y_gradXY
    gradDIV2_2D = FIELD_X_gradXY + FIELD_Y_gradYY
    gradDIV3_2D = np.zeros(gradDIV2_2D.shape) 


    rhs_eq1_2D = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV1_2D) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT1_2D)
    rhs_eq2_2D = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV2_2D) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT2_2D)
    rhs_eq3_2D = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV3_2D) + ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT3_2D)

    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,10),nrows=3)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Test 2D elastic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Test 2D elastic Wave Equation (unfiltered)')
    ax1.set_title('Difference at the order of '+str(abs(rhs_eq1_2D[posX,posY,posZ,:]-dttv1[posX,posY,posZ,:]).sum()))    
    ax1.plot(rhs_eq1_2D[posX,posY,posZ,:,nf], label='RHS elastic WE with FD gradients')
    ax1.plot(dttv1[posX,posY,posZ,:,nf], label='Temporal derivative of FIELDlacement', linestyle='-.')
    ax1.set_ylabel('X - component')
    ax1.legend()

    ax2.plot(rhs_eq2_2D[posX,posY,posZ,:,nf])
    ax2.plot(dttv2[posX,posY,posZ,:,nf], linestyle='-.')
    ax2.set_ylabel('Y - component')

    ax3.plot(rhs_eq3_2D[posX,posY,posZ,:,nf])
    ax3.plot(dttv3[posX,posY,posZ,:,nf], linestyle='-.')
    ax3.set_ylabel('Z - component')
    
    return rotROT1_2D, rotROT2_2D, rotROT3_2D, gradDIV1_2D, gradDIV2_2D, gradDIV3_2D


def WE_3D_ACOUSTIC(FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
                FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
                FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ,\
                vp_sub, vs_sub, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
    
    acc1 = FIELD_X_gradXX + FIELD_X_gradYY + FIELD_X_gradZZ 
    acc2 = FIELD_Y_gradXX + FIELD_Y_gradYY + FIELD_Y_gradZZ 
    acc3 = FIELD_Z_gradXX + FIELD_Z_gradYY + FIELD_Z_gradZZ 


    ## Check if they fit for true velocity model
    
    rhs_eq1 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) *acc1)
    rhs_eq2 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) *acc2)
    rhs_eq3 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) *acc3)
    
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,10),nrows=3)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Test 3D elastic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Test 3D elastic Wave Equation (unfiltered)')
    ax1.set_title('Difference at the order of '+str(abs(rhs_eq1[posX,posY,posZ,:]-dttv1[posX,posY,posZ,:]).sum()))
    ax1.plot(rhs_eq1[posX,posY,posZ,:,nf], label='RHS elastic WE with FD gradients')
    ax1.plot(dttv1[posX,posY,posZ,:,nf], label='Temporal derivative of displacement', linestyle='-.')
    ax1.set_ylabel('X - component')
    ax1.legend()

    ax2.plot(rhs_eq2[posX,posY,posZ,:,nf])
    ax2.plot(dttv2[posX,posY,posZ,:,nf], linestyle='-.')
    ax2.set_ylabel('Y - component')

    ax3.plot(rhs_eq3[posX,posY,posZ,:,nf])
    ax3.plot(dttv3[posX,posY,posZ,:,nf], linestyle='-.')
    ax3.set_ylabel('Z - component')

    return acc1, acc2, acc3


def WE_2D_ACOUSTIC(FIELD_X_gradXX, FIELD_Y_gradXX, FIELD_Z_gradXX, FIELD_X_gradYY, FIELD_Y_gradYY, FIELD_Z_gradYY, FIELD_X_gradZZ,\
                FIELD_Y_gradZZ, FIELD_Z_gradZZ, FIELD_X_gradXY, FIELD_Y_gradXY, FIELD_Z_gradXY, FIELD_X_gradYX, FIELD_Y_gradYX,\
                FIELD_Z_gradYX, FIELD_X_gradYZ, FIELD_Y_gradYZ, FIELD_Z_gradYZ, FIELD_X_gradXZ, FIELD_Y_gradXZ, FIELD_Z_gradXZ,\
                vp_sub, vs_sub, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered):
    
    acc1_2D = FIELD_X_gradXX + FIELD_X_gradYY 
    acc2_2D = FIELD_Y_gradXX + FIELD_Y_gradYY
    acc3_2D = FIELD_Z_gradXX + FIELD_Z_gradYY 


    ## Check if they fit for true velocity model
    
    rhs_eq1 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) *acc1_2D)
    rhs_eq2 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) *acc2_2D)
    rhs_eq3 = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) *acc3_2D)
    
    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,10),nrows=3)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Test 3D elastic Wave Equation at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Test 3D elastic Wave Equation (unfiltered)')
    ax1.set_title('Difference at the order of '+str(abs(rhs_eq1[posX,posY,posZ,:]-dttv1[posX,posY,posZ,:]).sum()))
    ax1.plot(rhs_eq1[posX,posY,posZ,:,nf], label='RHS elastic WE with FD gradients')
    ax1.plot(dttv1[posX,posY,posZ,:,nf], label='Temporal derivative of displacement', linestyle='-.')
    ax1.set_ylabel('X - component')
    ax1.legend()

    ax2.plot(rhs_eq2[posX,posY,posZ,:,nf])
    ax2.plot(dttv2[posX,posY,posZ,:,nf], linestyle='-.')
    ax2.set_ylabel('Y - component')

    ax3.plot(rhs_eq3[posX,posY,posZ,:,nf])
    ax3.plot(dttv3[posX,posY,posZ,:,nf], linestyle='-.')
    ax3.set_ylabel('Z - component')

    return acc1_2D, acc2_2D, acc3_2D
