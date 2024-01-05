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

from mpl_toolkits.axes_grid1 import make_axes_locatable, SubplotDivider
import matplotlib.pyplot as plt
import numpy as np


def image_model(path, field, field_sub, title, sx, sy, nx, ny, rx0, rx1, ry0, ry1, grid, nr,I, label_name):
    fig, (ax1,ax2) = plt.subplots(figsize=(12,10),ncols=2) #, sharey=True
    #divider = SubplotDivider(fig, 2, 1, 1, aspect=True)
    #plt.suptitle(r'Attenuation model $q_{kappa}$ at z=0', fontsize =16)
    plt.suptitle(r''+str(title), fontsize =16, y=0.75)
    ax1.set_title('Whole Model', fontsize =14)
    im = ax1.imshow(field, origin='lower', vmin=np.min(field), vmax=np.max(field))
    fig.subplots_adjust(wspace=0.5)
    ax1.scatter(sx,sy, color='purple', label = 'Source location')
    ax1.scatter(grid[0], grid[1], color='r', s=2, label= 'Receiver Grid')
    ax1.set_xlim([0,nx])
    ax1.set_ylim([0,ny])
    ax1.set_ylabel('y-axis [m]', fontsize=14)
    ax1.set_xlabel('x-axis [m]', fontsize=14)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label_name,size=18)
    ax1.legend()

    im2 = ax2.imshow(field_sub, origin='lower', vmin=np.min(field_sub), vmax=np.max(field_sub))
    #ax2.set_xlim([rx0,rx1])
    #ax2.set_ylim([ry0,ry1])
    ax2.set_ylabel('y-axis [m]', fontsize=14)
    ax2.set_xlabel('x-axis [m]', fontsize=14)
    ax2.set_title('Subsampled Receiver Field', fontsize =14, color='r')

    #plt.sca(ax2)
    #plt.xticks(np.arange(0,nr),np.round(np.linspace(rx0,rx1,nr),0))
    #ax2.set_xticks(np.round(np.linspace(rx0,rx1,5),0))
    ax2.set_xticks([0,nr-I])
    ax2.set_xticklabels([rx0+0.5,rx1-0.5], color='r')
    ax2.set_yticks([0,nr-I])
    ax2.set_yticklabels([ry0+0.5,ry1-0.5], color='r')


    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label(label_name,size=18)
    plt.savefig(str(path)+'/'+str(title))
    plt.show()


def image_field(path, field, field_sub, title, nr, rx0, rx1, ry0, ry1, text1, text2,I, label_name, colormap):
    fig, (ax1,ax2) = plt.subplots(figsize=(16,14),ncols=2) #, sharey=True
    plt.suptitle(r''+str(title), fontsize =16, y=0.75)
    ax1.set_title(text1, fontsize =14)
    im = ax1.imshow(field, origin='lower', vmin=np.min(field), vmax=np.max(field), cmap=colormap)
    fig.subplots_adjust(wspace=0.5)
    ax1.set_xticks([0,nr-I])
    ax1.set_xticklabels([rx0,rx1])
    ax1.set_yticks([0,nr-I])
    ax1.set_yticklabels([ry0, ry1])
    ax1.set_ylabel('y-axis [m]', fontsize=18)
    ax1.set_xlabel('x-axis [m]', fontsize=18)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, cax=cax,format='%.2f')
    cbar.set_label(label_name,size=18)

    im2 = ax2.imshow(field_sub, origin='lower', vmin=np.min(field_sub), vmax=np.max(field_sub), cmap=colormap)
    #ax2.set_ylabel('y-axis [m]', fontsize=14)
    ax2.set_xlabel('x-axis [m]', fontsize=18)
    ax2.set_title(text2, fontsize =18)
    ax2.set_xticks([0,nr-I])
    ax2.set_xticklabels([rx0,rx1])
    ax2.set_yticks([0,nr-I])
    ax2.set_yticklabels([ry0, ry1])


    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im2, cax=cax,format='%.2f')
    cbar.set_label(label_name,size=18)
    
    plt.savefig(str(path)+'/'+str(title))
    plt.show()


def depth_model(field, field_sub, title, field_name, rx0, rx1, ry0, ry1, pos1, pos2, ind1, ind2, grid, nr, nz):
    fig, (ax1,ax2) = plt.subplots(figsize=(12,6),ncols=2) #, sharey=True
    
    im1 = ax1.imshow(field_sub[:,:,-1], origin='lower', vmin=np.min(field_sub[:,:,-1]), vmax=np.max(field_sub[:,:,-1]))
    fig.subplots_adjust(wspace=0.5)
    plt.suptitle(r''+str(title), fontsize =16, y=0.98)
    ax1.set_ylabel('y-axis [m]', fontsize=14)
    ax1.set_xlabel('x-axis [m]', fontsize=14)
    ax1.set_title('Receiver Field  at z=0', fontsize =14)

    ax1.set_xticks([0,nr-1])
    ax1.set_xticklabels([rx0+0.5,rx1-0.5])
    ax1.set_yticks([0,nr-1])
    ax1.set_yticklabels([ry0+0.5, ry1-0.5])

    ax1.scatter(grid[0]-(rx0+0.5), grid[1]-(ry0+0.5), color='r', s=2, label= 'Receiver Grid')
    ax1.scatter(pos1-(rx0+0.5), pos2-(ry0+0.5), color='cyan', s=200, marker='*', label= 'Depth Profile Location')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cb = plt.colorbar(im1, cax=cax)
    cb.set_label(label=str(field_name), size=14)
    
    ax2.set_title('Depth model', fontsize =14)
    ax2.plot(field_sub[ind1,ind2,::-1], np.arange(0, nz))
    ax2.set_ylim([0,400])
    ax2.invert_yaxis()
    ax2.set_xlim([np.min(field),np.max(field)])
    ax2.set_ylabel('Depth [m]', fontsize=14)
    ax2.set_xlabel(str(field_name), fontsize=14)

    plt.show()



