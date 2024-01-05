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

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, SubplotDivider


def OBJ_FCT(theta1, theta2, comp, leng, nt_range, gradDIV1_internal, gradDIV2_internal, gradDIV3_internal,rotROT1_internal, \
            rotROT2_internal,rotROT3_internal, dttv1_internal, dttv2_internal, dttv3_internal, vp_sub_internal, vs_sub_internal,path,filtered, f_filt_cent,posX,posY,posZ):
    
    # specify position & time range in function input as such gradDIV3_internal[posX,posY,posZ,a:b]
    
    H_theta3 = np.zeros((leng, leng, nt_range))
    H_theta2 = np.zeros((leng, leng, nt_range))
    H_theta1 = np.zeros((leng, leng, nt_range))

    if comp =='X':
        lhs1         = dttv1_internal
        for jj in range(0, leng):   
            for mm in range(0, leng):
                H_theta1[jj,mm,:] = ((theta1[jj]**2) * gradDIV1_internal) - ((theta2[mm]**2)*rotROT1_internal)
        H_theta_sq  = (H_theta1 - (lhs1))**2
                    
    elif comp =='Y':
        lhs2         = dttv2_internal
        for jj in range(0, leng):   
            for mm in range(0, leng):
                H_theta2[jj,mm,:] = ((theta1[jj]**2) * gradDIV2_internal) - ((theta2[mm]**2)*rotROT2_internal)
        H_theta_sq  = (H_theta2 - (lhs2))**2
                
    elif comp =='Z':
        lhs3         = dttv3_internal
        for jj in range(0, leng):   
            for mm in range(0, leng):
                H_theta3[jj,mm,:] = ((theta1[jj]**2) * gradDIV3_internal) - ((theta2[mm]**2)*rotROT3_internal)
        H_theta_sq  = (H_theta3 - (lhs3))**2

    
    H_theta_sum = (1/(2*(nt_range)) * H_theta_sq.sum(2))
    yy          = np.log((H_theta_sum))
    
    mini = np.argwhere(np.round(yy,3)==np.min(np.round(yy,3)))
    true1 = np.argwhere(((theta1)<=(vp_sub_internal))) 
    true2 = np.argwhere(((theta2)<=(vs_sub_internal)))

    fig, (ax1,ax2) = plt.subplots(figsize=(24,12),ncols=2) 
    ax1.scatter(theta2[:],yy[0,:], color='blue', label = r'vp='+str(theta1[0]))
    ax1.scatter(theta2[:],yy[20,:], color='green', label = r'vp='+str(theta1[20]))
    ax1.scatter(theta2[:],yy[30,:], color='black', label = r'vp='+str(theta1[30]))
    ax1.scatter(theta2[:],yy[45,:], color='cyan', label = r'vp='+str(theta1[45]))
    ax1.scatter(theta2[:],yy[48,:], color='magenta', label = r'vp='+str(theta1[48]))
    ax1.scatter(theta2[:],yy[58,:], color='wheat', label = r'vp='+str(theta1[58]))
    ax1.scatter(theta2[:],yy[77,:], color='orange', label = r'vp='+str(theta1[77]))
    ax1.scatter(theta2[:],yy[144,:], color='red', label = r'vp='+str(theta1[144]))
    ax1.scatter(theta2[:],yy[175,:], color='pink', label = r'vp='+str(theta1[175]))
    #ax1.scatter(theta2[:],yy[true1[-1]+1,:],  color='chartreuse', label = r'vp='+str(theta1[true1[-1]+1][0]))
    ax1.scatter(theta2[:],yy[true1[-1],:],  color='chartreuse', label = r'vp='+str(theta1[true1[-1]][0]))
#     ax1.scatter(theta2[true2[-1]+1],yy[true1[-1]+1,true2[-1]+1], marker='*', color='black', s=200, label='True Solution')
    ax1.scatter(theta2[true2[-1]],yy[true1[-1],true2[-1]], marker='*', color='black', s=200, label='True Solution')
    ax1.scatter(theta2[:],yy[mini[0][0],:],  color='springgreen', label=r'vp='+str(theta1[mini[0][0]]))
    ax1.scatter(theta2[mini[0][1]],yy[mini[0][0],mini[0][1]], marker='*', color='yellow', s=200, label='Estimated Solution')
    ax1.set_xlabel('vs', fontsize=16)
    ax1.set_ylabel('Objective Function', fontsize=16)
    ax1.set_xlim(theta2[0],theta2[-1])
    ax1.legend(loc=1)
    
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    
    ax2.scatter(theta1[:],yy[:,0], color='blue', label = r'vs='+str(theta2[0]))
    ax2.scatter(theta1[:],yy[:,20], color='green', label = r'vs='+str(theta2[20]))
    ax2.scatter(theta1[:],yy[:,30], color='black', label = r'vs='+str(theta2[30]))
    ax2.scatter(theta1[:],yy[:,45], color='cyan', label = r'vs='+str(theta2[45]))
    ax2.scatter(theta1[:],yy[:,48], color='magenta', label = r'vs='+str(theta2[48]))
    ax2.scatter(theta1[:],yy[:,58], color='wheat', label = r'vs='+str(theta2[58]))
    ax2.scatter(theta1[:],yy[:,77], color='orange', label = r'vs='+str(theta2[77]))
    ax2.scatter(theta1[:],yy[:,144], color='red', label = r'vs='+str(theta2[144]))
    ax2.scatter(theta1[:],yy[:,175], color='pink', label = r'vs='+str(theta2[175]))
    #ax2.scatter(theta1[:],yy[:,true2[-1]+1], color='chartreuse', label = r'vs='+str(theta2[true2[-1]+1][0]))
    ax2.scatter(theta1[:],yy[:,true2[-1]], color='chartreuse', label = r'vs='+str(theta2[true2[-1]][0]))
    ax2.scatter(theta1[true1[-1]],yy[true1[-1],true2[-1]], marker='*', color='black', s=200, label='True Solution')
    #ax2.scatter(theta1[true1[-1]+1],yy[true1[-1]+1,true2[-1]+1], marker='*', color='black', s=200, label='True Solution')

    ax2.scatter(theta1[:],yy[:,mini[0][1]],  color='springgreen', label=r'vs='+str(theta2[mini[0][1]]))
    ax2.scatter(theta1[mini[0][0]],yy[mini[0][0],mini[0][1]], marker='*', color='yellow', s=200, label='Estimated Solution')
    ax2.set_xlabel('vp', fontsize=16)
    ax2.set_ylabel('Objective Function', fontsize=16)
    ax2.set_xlim(theta1[0],theta1[-1])
    ax2.legend(loc=1)
    if filtered=='yes':
        plt.savefig(path+str('/1Dobj_fct_f%d_x%d_y%d_z%d'%(f_filt_cent,posX,posY,posZ)))
    else:
        plt.savefig(path+str('/1Dobj_fct_unfiltered_x%d_y%d_z%d'%(posX,posY,posZ)))
    
    return mini,yy


def FCT_2D_OBJ(yy, theta1, theta2, mini, vp_sub_internal, vs_sub_internal, path, f_filt_cent, filtered, posX, posY, posZ):
    plt.figure(figsize=(10,8))
    plt.imshow((yy.T),origin='lower',extent=[theta1[0],theta1[-1], theta2[0],theta2[-1]])
    
    #if vel_model=='homogeneous':
    #    plt.scatter(vp_sub_internal, vs_sub_internal, label='True Homogeneous Model',s=100,marker='*')
    #    plt.scatter(theta1[mini[0][0]],theta2[mini[0][1]], color='crimson', label=r'vp,vs at min($J({\theta}))$')


    #if vel_model=='heterogeneous_z':
    #    plt.scatter(vp[-1],vs[-1], label='True Solution Layer 1',s=100,marker='*')
    #    plt.scatter(vp[0],vs[0], label='True Solution Layer 2',s=100,marker='*', color='lightgreen')#, color='blue')
    #    plt.scatter(theta1[mini[0][0]],theta2[mini[0][1]], color='crimson', label=r'vp,vs at min($J({\theta}))$')

    #plt.scatter(vp_sub[1,1,1],vs_sub[1,1,1], label='True Solution at FS ',s=100,marker='*')
    plt.scatter(vp_sub_internal, vs_sub_internal, label='True Solution',s=100,marker='*')
    plt.scatter(theta1[mini[0][0]],theta2[mini[0][1]], color='crimson', label=r'vp,vs at min($J({\theta}))$')


    plt.xlabel('vp velocity',fontsize=14)
    plt.ylabel('vs velocity',fontsize=14)
    plt.title(r'Objective function J$({\theta})$',fontsize=16)
    plt.colorbar(label=r'log[$J({\theta})$]')
    plt.legend(loc=2)
    plt.clim(np.min(yy.T), np.max(yy.T))
    if filtered=='yes':
        plt.savefig(path+str('/2Dobj_fct_f%d_x%d_y%d_z%d'%(f_filt_cent,posX,posY,posZ)))
    else:
        plt.savefig(path+str('/2Dobj_fct_unfiltered_x%d_y%d_z%d'%(posX,posY,posZ)))
    plt.show()

# +
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
# #%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

def FCT_3D_OBJ(tet1, tet2, yy, theta1, theta2, mini, path, posX, posY, posZ, f_filt_cent, filtered,vp_sub, vs_sub):
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111,projection='3d')
    im=ax.plot_surface(tet1,tet2,yy.T,  cmap='viridis',rcount=yy.shape[0], ccount=yy.shape[1], zorder=10)
    ax.scatter(theta1[mini[0][0]], theta2[mini[0][1]],yy[mini[0][0],mini[0][1]], color='red',)
    ax.scatter(vp_sub,vs_sub,yy[mini[0][0],mini[0][1]],marker='*', color='blue',)
    plt.xlabel('P-wave velocity',fontsize=14)
    plt.ylabel('S-wave velocity',fontsize=14)
    cbar = plt.colorbar(im)
    cbar.set_label(label=r'log[$J({\theta})$]',size=18)
    #plt.colorbar()
    if filtered=='yes':
        plt.savefig(path+str('/3Dobj_fct_f%d_x%d_y%d_z%d'%(f_filt_cent,posX,posY,posZ)))
    else:
        plt.savefig(path+str('/3Dobj_fct_unfiltered_x%d_y%d_z%d'%(posX,posY,posZ)))
    plt.show()



# +
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
# #%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

def FCT_3D_OBJ_ratio(tet1, tet2, yy, theta1, theta2, mini, path, posX, posY, posZ, f_filt_cent, filtered,vp_sub, vs_sub):
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111,projection='3d')
    im=ax.plot_surface(tet1,tet2,yy.T,  cmap='viridis',rcount=yy.shape[0], ccount=yy.shape[1], zorder=10)
    ax.scatter(theta1[mini[0][0]], theta2[mini[0][1]],yy[mini[0][0],mini[0][1]], color='red',)
    ax.scatter(vp_sub,vs_sub,yy[mini[0][0],mini[0][1]],marker='*', color='blue',)
    plt.ylabel('ratio P/S',fontsize=14)
    plt.xlabel('S-wave velocity',fontsize=14)
    cbar = plt.colorbar(im)
    cbar.set_label(label=r'log[$J({\theta})$]',size=18)
    #plt.colorbar()
    if filtered=='yes':
        plt.savefig(path+str('/3Dobj_fct_f%d_x%d_y%d_z%d'%(f_filt_cent,posX,posY,posZ)))
    else:
        plt.savefig(path+str('/3Dobj_fct_unfiltered_x%d_y%d_z%d'%(posX,posY,posZ)))
    plt.show()

