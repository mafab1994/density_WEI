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

# +
import sys

sys.path.append('/exports/csce/datastore/geos/users/s2110831/JUPYTER_old/GRADIOMETRY_Synthetic/Codebase(ii)-REWRITE-04-2021/Functions')
from FCT_GRADIOMETRY_good_loop import *


# -

def FCT_2D_OBJ_ACC(yy, theta1, theta2, mini, vp_sub_internal, vs_sub_internal, path, f_filt_cent, filtered, posX, posY, posZ):
    plt.figure(figsize=(10,8))
    plt.imshow((yy.T),origin='lower')#,extent=[theta1[0],theta1[-1], theta2[0],theta2[-1]])
    
    #if vel_model=='homogeneous':
    #    plt.scatter(vp_sub_internal, vs_sub_internal, label='True Homogeneous Model',s=100,marker='*')
    #    plt.scatter(theta1[mini[0][0]],theta2[mini[0][1]], color='crimson', label=r'vp,vs at min($J({\theta}))$')


    #if vel_model=='heterogeneous_z':
    #    plt.scatter(vp[-1],vs[-1], label='True Solution Layer 1',s=100,marker='*')
    #    plt.scatter(vp[0],vs[0], label='True Solution Layer 2',s=100,marker='*', color='lightgreen')#, color='blue')
    #    plt.scatter(theta1[mini[0][0]],theta2[mini[0][1]], color='crimson', label=r'vp,vs at min($J({\theta}))$')

    #plt.scatter(vp_sub[1,1,1],vs_sub[1,1,1], label='True Solution at FS ',s=100,marker='*')

    ##usually uncommented
    #plt.scatter(vp_sub_internal, vs_sub_internal, label='True Solution',s=100,marker='*')
    #plt.scatter(theta1[mini[0][0]],theta2[mini[0][1]], color='crimson', label=r'rho,c at min($J({\theta}))$')


    plt.xlabel('rho gradient',fontsize=14)
    plt.ylabel('phase velocity',fontsize=14)
    plt.title(r'Objective function J$({\theta})$',fontsize=16)
    plt.colorbar(label=r'log[$J({\theta})$]')
    plt.legend(loc=2)
    plt.clim(np.min(yy.T), np.max(yy.T))
#     if filtered=='yes':
#         plt.savefig(path+str('/2Dobj_fct_f%d_x%d_y%d_z%d'%(f_filt_cent,posX,posY,posZ)))
#     else:
#         plt.savefig(path+str('/2Dobj_fct_unfiltered_x%d_y%d_z%d'%(posX,posY,posZ)))
#     plt.show()

# +
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
# #%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

def FCT_3D_OBJ_ACC(tet1, tet2, yy, theta1, theta2, mini, path, posX, posY, posZ, f_filt_cent, filtered,vp_sub, vs_sub):
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111,projection='3d')
    im=ax.plot_surface(tet1,tet2,yy.T,  cmap='viridis',rcount=yy.shape[0], ccount=yy.shape[1], zorder=10)
    #ax.scatter(theta1[mini[0][0]], theta2[mini[0][1]],yy[mini[0][0],mini[0][1]], color='red',)
    #ax.scatter(vp_sub,vs_sub,yy[mini[0][0],mini[0][1]],marker='*', color='blue',)
    plt.xlabel('tet1',fontsize=14)
    plt.ylabel('tet2',fontsize=14)
    cbar = plt.colorbar(im)
    cbar.set_label(label=r'log[$J({\theta})$]',size=18)
    #plt.colorbar()
#     if filtered=='yes':
#         plt.savefig(path+str('/3Dobj_fct_f%d_x%d_y%d_z%d'%(f_filt_cent,posX,posY,posZ)))
#     else:
#         plt.savefig(path+str('/3Dobj_fct_unfiltered_x%d_y%d_z%d'%(posX,posY,posZ)))
#     plt.show()

