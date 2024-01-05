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


def TRUE_EST_3D_ELASTIC(rotROT1, rotROT2, rotROT3, gradDIV1, gradDIV2, gradDIV3,\
                vp_sub, vs_sub, vp_est, vs_est, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered, path, t_sub):
    
    
    ## Check if they fit for true velocity model
    
    rhs_eq1_TRUE = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV1) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT1)
    rhs_eq2_TRUE = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV2) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT2)
    rhs_eq3_TRUE = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV3) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT3)
    
    rhs_eq1_EST = ((vp_est[:,:,:,nf, np.newaxis, np.newaxis]**2) * gradDIV1) - ((vs_est[:,:,:,nf, np.newaxis, np.newaxis]**2) *rotROT1)
    rhs_eq2_EST = ((vp_est[:,:,:,nf, np.newaxis, np.newaxis]**2) * gradDIV2) - ((vs_est[:,:,:,nf, np.newaxis, np.newaxis]**2) *rotROT2)
    rhs_eq3_EST = ((vp_est[:,:,:,nf, np.newaxis, np.newaxis]**2) * gradDIV3) - ((vs_est[:,:,:,nf, np.newaxis, np.newaxis]**2) *rotROT3)
    

    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,10),nrows=3)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Misfit at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Misfit (unfiltered)')
    #ax1.set_title('Difference at the order of '+str(abs(rhs_eq1_TRUE[posX,posY,posZ,:]-dttv1[posX,posY,posZ,:]).sum()))
    ax1.plot(t_sub,dttv1[posX,posY,posZ,:,nf], label='Temporal derivative of displacement', linestyle='-.')
    ax1.plot(t_sub,rhs_eq1_TRUE[posX,posY,posZ,:,nf], label='rhs true')
    ax1.plot(t_sub,rhs_eq1_EST[posX,posY,posZ,:,nf], label='rhs estimated', linestyle=':')
    ax1.set_ylabel('X - component')
    ax1.legend()
    
    ax2.plot(t_sub,dttv2[posX,posY,posZ,:,nf], linestyle='-.')
    ax2.plot(t_sub,rhs_eq2_TRUE[posX,posY,posZ,:,nf])
    ax2.plot(t_sub,rhs_eq2_EST[posX,posY,posZ,:,nf], linestyle=':')
    ax2.set_ylabel('Y - component')
    
    ax3.plot(t_sub,dttv3[posX,posY,posZ,:,nf],  linestyle='-.')
    ax3.plot(t_sub,rhs_eq3_TRUE[posX,posY,posZ,:,nf])
    ax3.plot(t_sub,rhs_eq3_EST[posX,posY,posZ,:,nf],  linestyle=':')
    ax3.set_ylabel('Z - component')
    ax3.set_xlabel('Time [s]')
    
    if filtered=='yes':
        plt.savefig(path+str('/TRACE_MISFIT_fct_f%d_x%d_y%d_z%d'%(f_filt_cent,posX,posY,posZ)))
    else:
        plt.savefig(path+str('/TRACE_MISFIT_fct_unfiltered_x%d_y%d_z%d'%(posX,posY,posZ)))
   
    return 


def MISFIT_3D_ELASTIC(rotROT1, rotROT2, rotROT3, gradDIV1, gradDIV2, gradDIV3,\
                vp_sub, vs_sub, vp_est, vs_est, dttv1, dttv2, dttv3, posX,posY, posZ, nf, f_filt_cent, filtered, path, t_sub):
    
    
    ## Check if they fit for true velocity model
    
    rhs_eq1_TRUE = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV1) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT1)
    rhs_eq2_TRUE = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV2) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT2)
    rhs_eq3_TRUE = ((vp_sub[:,:,:, np.newaxis, np.newaxis]**2) * gradDIV3) - ((vs_sub[:,:,:, np.newaxis, np.newaxis]**2) *rotROT3)
    
    rhs_eq1_EST = ((vp_est[:,:,:,nf, np.newaxis, np.newaxis]**2) * gradDIV1) - ((vs_est[:,:,:,nf, np.newaxis, np.newaxis]**2) *rotROT1)
    rhs_eq2_EST = ((vp_est[:,:,:,nf, np.newaxis, np.newaxis]**2) * gradDIV2) - ((vs_est[:,:,:,nf, np.newaxis, np.newaxis]**2) *rotROT2)
    rhs_eq3_EST = ((vp_est[:,:,:,nf, np.newaxis, np.newaxis]**2) * gradDIV3) - ((vs_est[:,:,:,nf, np.newaxis, np.newaxis]**2) *rotROT3)
    
    MISFIT1 = ((dttv1[posX,posY,:,:,nf]-rhs_eq1_EST[posX,posY,:,:,nf])**2)#.sum(1) / dttv1[posX,posY,:,:,nf].shape[1]
    MISFIT1_TRUE = ((dttv1[posX,posY,:,:,nf]-rhs_eq1_TRUE[posX,posY,:,:,nf])**2)#.sum(1)  / dttv1[posX,posY,:,:,nf].shape[1]
    MISFIT2 = ((dttv2[posX,posY,:,:,nf]-rhs_eq2_EST[posX,posY,:,:,nf])**2)#.sum(1) / dttv1[posX,posY,:,:,nf].shape[1]
    MISFIT2_TRUE = ((dttv2[posX,posY,:,:,nf]-rhs_eq2_TRUE[posX,posY,:,:,nf])**2)#.sum(1)  / dttv1[posX,posY,:,:,nf].shape[1]
    MISFIT3_TRUE = ((dttv3[posX,posY,:,:,nf]-rhs_eq3_TRUE[posX,posY,:,:,nf])**2)#.sum(1)  / dttv1[posX,posY,:,:,nf].shape[1]
    MISFIT3 = ((dttv3[posX,posY,:,:,nf]-rhs_eq3_EST[posX,posY,:,:,nf])**2)#.sum(1)  / dttv1[posX,posY,:,:,nf].shape[1]


    fig, (ax1,ax2,ax3) = plt.subplots(figsize=(12,10),nrows=3)
    #fig.subplots_adjust(wspace=50)
    if filtered =='yes':
        plt.suptitle('Misfit at '+str(f_filt_cent[nf])+' [Hz]')
    else:
        plt.suptitle('Misfit (unfiltered)')
    #ax1.set_title('Difference at the order of '+str(abs(rhs_eq1_TRUE[posX,posY,posZ,:]-dttv1[posX,posY,posZ,:]).sum()))
    ax1.plot(t_sub,MISFIT1[posZ,:], label='misfit estimated', linestyle=':')
    ax1.plot(t_sub, MISFIT1_TRUE[posZ,:], label='misfit true')

    #ax1.plot(dttv1[posX,posY,posZ,:,nf], label='Temporal derivative of displacement', linestyle='-.')
    ax1.set_ylabel('X - component')
    ax1.legend()
    
    ax2.plot(t_sub,MISFIT2[posZ,:], linestyle=':')
    ax2.plot(t_sub,MISFIT2_TRUE[posZ,:])
    #ax2.plot(dttv2[posX,posY,posZ,:,nf], linestyle='-.')
    ax2.set_ylabel('Y - component')
    
    ax3.plot(t_sub,MISFIT3[posZ,:], linestyle=':')
    ax3.plot(t_sub,MISFIT3_TRUE[posZ,:])
    #ax3.plot(dttv3[posX,posY,posZ,:,nf], linestyle='-.')
    ax3.set_ylabel('Z - component')
    ax3.set_xlabel('Time [s]')
    
    SUM_MISFIT1 = MISFIT1.sum(1) / dttv1[posX,posY,:,:,nf].shape[1]
    SUM_MISFIT1_TRUE = MISFIT1_TRUE.sum(1)  / dttv1[posX,posY,:,:,nf].shape[1]
    SUM_MISFIT2 = MISFIT2.sum(1) / dttv1[posX,posY,:,:,nf].shape[1]
    SUM_MISFIT2_TRUE = MISFIT2_TRUE.sum(1)  / dttv1[posX,posY,:,:,nf].shape[1]
    SUM_MISFIT3_TRUE = MISFIT3_TRUE.sum(1)  / dttv1[posX,posY,:,:,nf].shape[1]
    SUM_MISFIT3 = MISFIT3.sum(1)  / dttv1[posX,posY,:,:,nf].shape[1]
    
    if filtered=='yes':
        plt.savefig(path+str('/MISFIT_fct_f%d_x%d_y%d_z%d'%(f_filt_cent,posX,posY,posZ)))
    else:
        plt.savefig(path+str('/MISFIT_fct_unfiltered_x%d_y%d_z%d'%(posX,posY,posZ)))
        
    return rhs_eq1_TRUE, rhs_eq2_TRUE, rhs_eq3_TRUE, rhs_eq1_EST, rhs_eq2_EST, rhs_eq3_EST, MISFIT1, MISFIT2, MISFIT3, \
            MISFIT1_TRUE, MISFIT2_TRUE, MISFIT3_TRUE, SUM_MISFIT1, SUM_MISFIT2, SUM_MISFIT3, \
            SUM_MISFIT1_TRUE, SUM_MISFIT2_TRUE, SUM_MISFIT3_TRUE


