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
from scipy import signal
import matplotlib.pyplot as plt 
import numpy as np
#import dask

#@dask.delayed
def BP_filter(U_field, f_BAND, dt_sub, pos_x, pos_z, t_sub):
    sos=np.zeros((5,6,len(f_BAND[0,:])))
    #U_filter = np.zeros((U_field.shape[0],U_field.shape[1],U_field.shape[2],len(f_BAND[0,:])))
    #for ii in range(0,len(f_BAND[0,:])):
    #    sos[:,:,ii] = signal.butter(5, f_BAND[:,ii], 'bandpass', fs=1/dt_sub, output='sos' ) #higher order ?
    #    U_filter[:,:,:,ii] = signal.sosfilt(np.ascontiguousarray(sos[:,:,ii]), U_field) # use sosfiltfilt for correct phase??
    #6
    sos = [signal.butter(9, np.array([f_BAND[0][i], f_BAND[1][i]]), 'bandpass', fs=1/dt_sub, output='sos' ) for i in range(0,len(f_BAND[0])) ]
    sos = np.array(sos)
    U_filter = [signal.sosfilt(np.ascontiguousarray(jj), U_field) for jj in sos]
    U_filter = np.array(U_filter)
    
    f_filt_cent = (f_BAND[1,:]+f_BAND[0,:])/2
     
    
    return U_filter, f_filt_cent
