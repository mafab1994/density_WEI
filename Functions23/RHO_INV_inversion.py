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
import cmcrameri as cmm
import numpy as np
import matplotlib.pyplot as plt

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    #plt.show()
    
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

def rho_inverse0(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it):
    
    #mean = np.mean(utt_g_st,axis=2)
    #var = (utt_g_st-mean[:,:,np.newaxis])**2
    
    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    K = c_input**2 *m_init  
    beta = 1/(K)

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2 #/ var[:,:,i].flatten()  
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = np.diag((a_s**2).flatten())#I*(a_s**2)

    for n in range(0,nt_sub):

        G =  WW*(M_l[:,:,n]) #/ var[:,:,n]
        G_t = np.matrix.getH(G)

#         A[:,:,n]     = (np.dot(G_t,G)/ var[:,:,n]) + damp
        A[:,:,n]    = np.dot(G_t,G) + damp 
        bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 
    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    rho_inv                   = np.reshape(1/rr, (nr,nr))    
    
        
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
#     plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].flatten()[::-1])),1/m_true[1:-1,1:-1].flatten()[::-1], label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
#     plt.plot(1/m_true[1:-1,1:-1].flatten()[::-1], linewidth=2, color='tab:blue')
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].T.flatten())),1/m_true[1:-1,1:-1].T.flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].T.flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
#     plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].flatten()[::-1])),1/rrr[1:-1,1:-1].flatten()[::-1], label='Inversion result',s=40,marker='+', color='coral')
#     plt.plot(1/rrr[1:-1,1:-1].flatten()[::-1],color='coral')
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),1/rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(1/rrr[1:-1,1:-1].T.flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    #plt.show()
    
    m = m_true.T.flatten()
    r=rr.T.flatten()#[::-1]
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    #plt.show()
    
    
    #rho_inv = np.reshape(1/rr, (nr,nr))[::-1,::-1]
    #rho_inv = rho_inv[::-1,::-1]
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    #plt.xlabel('X-axis',fontsize=16)
    #plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('True Density Model')
    im_true=plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    fig = plt.figure(figsize=(8,6))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] ,vmin=np.min((diff[1:-1,1:-1])), vmax=np.max((diff[1:-1,1:-1])), aspect='auto', interpolation='none',origin='lower', cmap=colormap,extent=[0.5, 8.5, 0.5, 8.5])
    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(label_name,size=18)
#     plt.xlabel('X-axis',fontsize=16)
#     plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    
    return rho_inv, rr, misfit_inv, RMS, difff, diff



# +
import cmcrameri as cmm
import numpy as np
import matplotlib.pyplot as plt

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    #plt.show()
    
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

def rho_inverse0_dK(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it):

    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    K = c_input**2 *m_init  
    beta = 1/c_input**2

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = np.diag((a_s**2).flatten())#I*(a_s**2)

    for n in range(0,nt_sub):

        G =  WW*(M_l[:,:,n]) 
        G_t = np.matrix.getH(G)

        A[:,:,n]     = np.dot(G_t,G) + damp  
        bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 
    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    rho_inv                   = np.reshape(1/rr, (nr,nr))    
    
        
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
#     plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].flatten()[::-1])),1/m_true[1:-1,1:-1].flatten()[::-1], label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
#     plt.plot(1/m_true[1:-1,1:-1].flatten()[::-1], linewidth=2, color='tab:blue')
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].T.flatten())),1/m_true[1:-1,1:-1].T.flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].T.flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
#     plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].flatten()[::-1])),1/rrr[1:-1,1:-1].flatten()[::-1], label='Inversion result',s=40,marker='+', color='coral')
#     plt.plot(1/rrr[1:-1,1:-1].flatten()[::-1],color='coral')
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),1/rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(1/rrr[1:-1,1:-1].T.flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    #plt.show()
    
    m = m_true.T.flatten()
    r=rr.T.flatten()#[::-1]
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    #plt.show()
    
    
    #rho_inv = np.reshape(1/rr, (nr,nr))[::-1,::-1]
    #rho_inv = rho_inv[::-1,::-1]
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    #plt.xlabel('X-axis',fontsize=16)
    #plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('True Density Model')
    im_true=plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    fig = plt.figure(figsize=(8,6))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] ,vmin=np.min((diff[1:-1,1:-1])), vmax=np.max((diff[1:-1,1:-1])), aspect='auto', interpolation='none',origin='lower', cmap=colormap,extent=[0.5, 8.5, 0.5, 8.5])
    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(label_name,size=18)
#     plt.xlabel('X-axis',fontsize=16)
#     plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    
    return rho_inv, rr, misfit_inv, RMS, difff, diff


# +
import cmcrameri as cmm
import numpy as np
import matplotlib.pyplot as plt

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    #plt.show()
    
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

def rho_inverse0_elastic(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it):

    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    #K = c_input**2 * m_init
    K = c_input**2 * 1/m_init 
    beta = 1/(K)

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = np.diag((a_s**2).flatten())#I*(a_s**2)

    for n in range(0,nt_sub):

        G =  WW*(M_l[:,:,n]) 
        G_t = np.matrix.getH(G)

        A[:,:,n]     = np.dot(G_t,G) + damp  
        #bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 
        bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,m_init.flatten())) 
    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    #rho_inv                   = np.reshape(1/rr, (nr,nr))    
    rho_inv                   = np.reshape(rr, (nr,nr))    
    
        
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
#     plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].flatten()[::-1])),1/m_true[1:-1,1:-1].flatten()[::-1], label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
#     plt.plot(1/m_true[1:-1,1:-1].flatten()[::-1], linewidth=2, color='tab:blue')
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].T.flatten())),1/m_true[1:-1,1:-1].T.flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].T.flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
#     plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].flatten()[::-1])),1/rrr[1:-1,1:-1].flatten()[::-1], label='Inversion result',s=40,marker='+', color='coral')
#     plt.plot(1/rrr[1:-1,1:-1].flatten()[::-1],color='coral')
#     plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),1/rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
#     plt.plot(1/rrr[1:-1,1:-1].T.flatten(),color='coral')
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(rrr[1:-1,1:-1].T.flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    #plt.show()
    
    m = m_true.T.flatten()
    r=rr.T.flatten()#[::-1]
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    #plt.show()
    
    
    #rho_inv = np.reshape(1/rr, (nr,nr))[::-1,::-1]
    #rho_inv = rho_inv[::-1,::-1]
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    #plt.xlabel('X-axis',fontsize=16)
    #plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('True Density Model')
    im_true=plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    fig = plt.figure(figsize=(8,6))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] ,vmin=np.min((diff[1:-1,1:-1])), vmax=np.max((diff[1:-1,1:-1])), aspect='auto', interpolation='none',origin='lower', cmap=colormap,extent=[0.5, 8.5, 0.5, 8.5])
    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(label_name,size=18)
#     plt.xlabel('X-axis',fontsize=16)
#     plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    
    return rho_inv, rr, misfit_inv, RMS, difff, diff


# +
import cmcrameri as cmm
import numpy as np
import matplotlib.pyplot as plt

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    #plt.show()
    
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

def rho_inverse0WITHNOISE(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it,ERR):

    # True density model
    m_true = 1/rho_true

    # Build data vector
    
    b      = np.zeros((N,nt_sub))
    K = c_input**2 *m_init  
    beta = 1/(K)

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = np.diag((a_s**2).flatten())#I*(a_s**2)

    for n in range(0,nt_sub):
        ErrM = np.diag((ERR[:,:,n]).flatten())
        ErrM_t = np.matrix.getH(ErrM)
        
        G =  WW*(M_l[:,:,n]) 
        G_t = np.matrix.getH(G)
        
        err         = np.dot(ErrM_t,ErrM)
        err2        = np.dot(G_t,err)
        A[:,:,n]     = np.dot(err,G) + damp  
        bbb[:,n]    = np.dot(err2,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 
    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    rho_inv                   = np.reshape(1/rr, (nr,nr))    
    
        
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
#     plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].flatten()[::-1])),1/m_true[1:-1,1:-1].flatten()[::-1], label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
#     plt.plot(1/m_true[1:-1,1:-1].flatten()[::-1], linewidth=2, color='tab:blue')
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].T.flatten())),1/m_true[1:-1,1:-1].T.flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].T.flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
#     plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].flatten()[::-1])),1/rrr[1:-1,1:-1].flatten()[::-1], label='Inversion result',s=40,marker='+', color='coral')
#     plt.plot(1/rrr[1:-1,1:-1].flatten()[::-1],color='coral')
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),1/rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(1/rrr[1:-1,1:-1].T.flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    #plt.show()
    
    m = m_true.T.flatten()
    r=rr.T.flatten()#[::-1]
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    #plt.show()
    
    
    #rho_inv = np.reshape(1/rr, (nr,nr))[::-1,::-1]
    #rho_inv = rho_inv[::-1,::-1]
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    #plt.xlabel('X-axis',fontsize=16)
    #plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('True Density Model')
    im_true=plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    fig = plt.figure(figsize=(8,6))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] ,vmin=np.min((diff[1:-1,1:-1])), vmax=np.max((diff[1:-1,1:-1])), aspect='auto', interpolation='none',origin='lower', cmap=colormap,extent=[0.5, 8.5, 0.5, 8.5])
    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(label_name,size=18)
#     plt.xlabel('X-axis',fontsize=16)
#     plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    
    return rho_inv, rr, misfit_inv, RMS, difff, diff


# +
import numpy as np
import matplotlib.pyplot as plt

def rho_inverse0_3D(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it, nrz):

    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    K = c_input**2 *m_init  
    beta = 1/(K)

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,:,i].flatten()) * (dist_r**2) *2
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = I*a_s**2

    for n in range(0,nt_sub):

        G =  WW*(M_l[:,:,n]) 
        G_t = np.matrix.getH(G)

        A[:,:,n]     = np.dot(G_t,G) + damp  
        bbb[:,n]    = np.dot(G_t,b[:,n])+ (a_s**2 * np.dot(I,1/m_init.flatten())) 
    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    rho_inv                   = np.reshape(1/rr, (nr,nr,nrz))    
        
        
        
    
    
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1,1].flatten())),1/m_true[1:-1,1:-1,1].flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1,1].flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr,nrz))
    rrr = rrr[:,:,1]
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].flatten())),1/rrr[1:-1,1:-1].flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(1/rrr[1:-1,1:-1].flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    #plt.show()
    
    m = m_true[1:-1,1:-1,1].flatten()
    r=rrr[:,:].flatten()
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    #plt.show()
    
    
    rho_inv = np.reshape(1/rrr, (nr,nr))
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap='pink_r',extent=[1, 8, 1, 8])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('True Density Model')
    plt.imshow(rho_true[1:-1,1:-1,1] ,vmin=np.min(rho_true[1:-1,1:-1,1]), vmax=np.max(rho_true[1:-1,1:-1,1]), aspect='auto', interpolation='none' ,origin='lower', cmap='pink_r',extent=[1, 8, 1, 8]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    #plt.colorbar()     
    #plt.show() 
    
    diff = 100*(rho_inv-rho_true[:,:,1])/rho_true[:,:,1]
    difff = (rho_true[:,:,1]-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    fig = plt.figure(figsize=(8,6))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] ,vmin=np.min((diff[1:-1,1:-1])), vmax=np.max((diff[1:-1,1:-1])), aspect='auto', interpolation='none',origin='lower', cmap=colormap,extent=[1, 8, 1, 8])
    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(label_name,size=18)
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1,1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    
    rho_inv = np.reshape(1/rr, (nr,nr,nrz))
    return rho_inv, rr, misfit_inv, RMS, difff, diff


# +
import cmcrameri as cmm
import numpy as np
import matplotlib.pyplot as plt

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()
    
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)


def rho_inverse(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it,maxMIS, minMIS):

    #mean = np.mean(utt_g_st,axis=2)
    #var = (utt_g_st-mean[:,:,np.newaxis])**2
    
    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))
    K = c_input**2 * m_init
    beta = 1/(K)

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2 #/ var[:,:,i].flatten()
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))
    rho_inv_t = np.zeros((nr,nr,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = np.diag((a_s**2).flatten())#I*a_s**2

    for n in range(0,nt_sub):

        G =  WW*(M_l[:,:,n]) 
        G_t = np.matrix.getH(G)

#         A[:,:,n]     = (np.dot(G_t,G)/ var[:,:,n]) + damp 
        A[:,:,n]    = np.dot(G_t,G) + damp 
        bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 

#         rr[:,n]                       = np.dot(np.linalg.inv(A[:,:,n]), bbb[:,n])#np.linalg.solve(M_l.sum(2),dd)
#         rho_inv_t[:,:,n]                   = np.reshape(1/rr[:,n] , (nr,nr))    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    rho_inv                   = np.reshape(1/rr, (nr,nr))  
    
 
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].T.flatten())),1/m_true[1:-1,1:-1].T.flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].T.flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),1/rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(1/rrr[1:-1,1:-1].T.flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    #plt.show()
    
    m = m_true.T.flatten()
    r=rrr.T.flatten()#rr.T.flatten()
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    #plt.show()
    
    
    #rho_inv = np.reshape(1/rr, (nr,nr))
    #rho_inv = rho_inv[::-1,::-1]
    fig,ax = plt.subplots(figsize=(10,8))

    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    #cbar = plt.colorbar(im, format='%.0f')
    #cbar.set_label(label_name,size=18)
    ##plt.xlabel('X-axis',fontsize=16)
    #plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('True Density Model')
    im_true = plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar.set_label(label_name,size=18)
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    #im=plt.imshow(diff[1:-1,1:-1] ,vmin=minMIS, vmax=maxMIS, aspect='auto', interpolation='none',origin='lower', cmap='bwr',extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])
    im=plt.imshow(diff[1:-1,1:-1] , aspect='auto', interpolation='none',origin='lower', cmap='bwr',extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])

    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    #cbar.set_label(label_name,size=18)
#     plt.xlabel('X-axis',fontsize=16)
#     plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    #rho_inv = rho_inv[::-1,::-1]
    return rho_inv, rr, misfit_inv, RMS, difff, diff#, b,bbb,dd#, rho_inv_t#, cost


# +
import cmcrameri as cmm
import numpy as np
import matplotlib.pyplot as plt

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()
    
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)


def rho_inverse_dK(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it,maxMIS, minMIS):

    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))
    K = c_input**2 * m_init
    beta = 1/c_input**2

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))
    rho_inv_t = np.zeros((nr,nr,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = np.diag((a_s**2).flatten())#I*a_s**2

    for n in range(0,nt_sub):

        G =  WW*(M_l[:,:,n]) 
        G_t = np.matrix.getH(G)

        A[:,:,n]     = np.dot(G_t,G) + damp  
        bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 

#         rr[:,n]                       = np.dot(np.linalg.inv(A[:,:,n]), bbb[:,n])#np.linalg.solve(M_l.sum(2),dd)
#         rho_inv_t[:,:,n]                   = np.reshape(1/rr[:,n] , (nr,nr))    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    rho_inv                   = np.reshape(1/rr, (nr,nr))  
    
 
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].T.flatten())),1/m_true[1:-1,1:-1].T.flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].T.flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),1/rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(1/rrr[1:-1,1:-1].T.flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    #plt.show()
    
    m = m_true.T.flatten()
    r=rrr.T.flatten()#rr.T.flatten()
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    #plt.show()
    
    
    #rho_inv = np.reshape(1/rr, (nr,nr))
    #rho_inv = rho_inv[::-1,::-1]
    fig,ax = plt.subplots(figsize=(10,8))

    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    #cbar = plt.colorbar(im, format='%.0f')
    #cbar.set_label(label_name,size=18)
    ##plt.xlabel('X-axis',fontsize=16)
    #plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('True Density Model')
    im_true = plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar.set_label(label_name,size=18)
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    #im=plt.imshow(diff[1:-1,1:-1] ,vmin=minMIS, vmax=maxMIS, aspect='auto', interpolation='none',origin='lower', cmap='bwr',extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])
    im=plt.imshow(diff[1:-1,1:-1] , aspect='auto', interpolation='none',origin='lower', cmap='bwr',extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])

    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    #cbar.set_label(label_name,size=18)
#     plt.xlabel('X-axis',fontsize=16)
#     plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    #rho_inv = rho_inv[::-1,::-1]
    return rho_inv, rr, misfit_inv, RMS, difff, diff#, rho_inv_t#, cost

# +
import cmcrameri as cmm
import numpy as np
import matplotlib.pyplot as plt

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()
    
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)


def rho_inverse_elastic(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it,maxMIS, minMIS):

    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))
    #K = c_input**2 * m_init
    K = c_input**2 * 1/m_init  
    beta = 1/(K)

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))
    rho_inv_t = np.zeros((nr,nr,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = np.diag((a_s**2).flatten())#I*a_s**2

    for n in range(0,nt_sub):

        G =  WW*(M_l[:,:,n]) 
        G_t = np.matrix.getH(G)

        A[:,:,n]     = np.dot(G_t,G) + damp  
        #bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 
        bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,m_init.flatten())) 

#         rr[:,n]                       = np.dot(np.linalg.inv(A[:,:,n]), bbb[:,n])#np.linalg.solve(M_l.sum(2),dd)
#         rho_inv_t[:,:,n]                   = np.reshape(1/rr[:,n] , (nr,nr))    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    #rho_inv                   = np.reshape(1/rr, (nr,nr))  
    rho_inv                   = np.reshape(rr, (nr,nr))    
    
 
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].T.flatten())),1/m_true[1:-1,1:-1].T.flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].T.flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
#     plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),1/rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
#     plt.plot(1/rrr[1:-1,1:-1].T.flatten(),color='coral')

    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].T.flatten())),rrr[1:-1,1:-1].T.flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(rrr[1:-1,1:-1].T.flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    #plt.show()
    
    m = m_true.T.flatten()
    r=rrr.T.flatten()#rr.T.flatten()
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.scatter(np.arange(0,len(r[begin:stop])),1/m[begin:stop], label = 'True model')
    #plt.plot(1/r[begin:stop], label='Inversion result')
    plt.plot(r[begin:stop], label='Inversion result')
    plt.scatter(np.arange(0,len(r[begin:stop])),r[begin:stop], label='Inversion result')

    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    #plt.show()
    
    
    #rho_inv = np.reshape(1/rr, (nr,nr))
    #rho_inv = rho_inv[::-1,::-1]
    fig,ax = plt.subplots(figsize=(10,8))

    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    #cbar = plt.colorbar(im, format='%.0f')
    #cbar.set_label(label_name,size=18)
    ##plt.xlabel('X-axis',fontsize=16)
    #plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('True Density Model')
    im_true = plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar.set_label(label_name,size=18)
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    #im=plt.imshow(diff[1:-1,1:-1] ,vmin=minMIS, vmax=maxMIS, aspect='auto', interpolation='none',origin='lower', cmap='bwr',extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])
    im=plt.imshow(diff[1:-1,1:-1] , aspect='auto', interpolation='none',origin='lower', cmap='bwr',extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])

    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    #cbar.set_label(label_name,size=18)
#     plt.xlabel('X-axis',fontsize=16)
#     plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label=r'$[\frac{g}{cm^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    #rho_inv = rho_inv[::-1,::-1]
    return rho_inv, rr, misfit_inv, RMS, difff, diff#, rho_inv_t#, cost

# -


def costFCT():
    I          = np.eye(N, N)
    damp       = I*a_s**2

    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))
    cost_t = np.zeros((N,nt_sub))

    b      = np.zeros((N,nt_sub))
    K = c_input**2 *m_init  
    beta = 1/(K)

    for n in range(nt_sub):

            b[:,n] = (beta.flatten()*time_field[:,:,0,n,0].flatten()) * (dist_r**2) *2

            G =  WW*(M_l[:,:,n]) 
            G_t = np.matrix.getH(G)

            A[:,:,n]     = np.dot(G_t,G) + damp  
            bbb[:,n]    = np.dot(G_t,b[:,n])+ (a_s**2 * np.dot(I,1/m_init.flatten())) 

            cost_t[:,n] = (1/nt_sub) *( (np.dot(G_t,b[:,n]) - np.dot(A[:,:,n],1/rho_inv2.flatten()))**2 ).sum()# + (a_s * ((rho_inv2.flatten()-(m_init.flatten()) )**2))

    cost = cost_t.sum(1)
    
    return cost


# +
import numpy as np
import matplotlib.pyplot as plt

def rho_inverse3C(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_lx, Mly, Mlz,nr, begin,stop,test,a_s, n_it,maxMIS, minMIS):

    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    K = c_input**2 *m_init  
    beta = 1/(K)

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = I*a_s**2

    for n in range(0,nt_sub):

        Gx =  WW*(M_lx[:,:,n]) 
        Gy =  WW*(M_ly[:,:,n]) 
        Gz =  WW*(M_lz[:,:,n]) 
        G=[]
        G= np.append(Gx,Gy)
        G= np.append(G,Gz)
        G = np.reshape(np.asarray(G), (M_lx[:,:,n].shape))
        
        G_t = np.matrix.getH(G)

        A[:,:,n]     = np.dot(G_t,G) + damp  
        bbb[:,n]    = np.dot(G_t,b[:,n])+ (a_s**2 * np.dot(I,1/m_init.flatten())) 
    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)#np.linalg.solve(M_l.sum(2),dd)
    rho_inv                   = np.reshape(1/rr, (nr,nr))    
        
        
        
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].flatten())),1/m_true[1:-1,1:-1].flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].flatten())),1/rrr[1:-1,1:-1].flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(1/rrr[1:-1,1:-1].flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    plt.show()
    
    m = m_true.flatten()
    r=rr.flatten()
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    plt.show()
    
    
    rho_inv = np.reshape(1/rr, (nr,nr))
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap='pink_r')#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    #plt.colorbar()     
    plt.show() 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('True Density Model')
    plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap='pink_r') #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    #plt.colorbar()     
    plt.show() 
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    fig = plt.figure(figsize=(8,6))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] ,vmin=minMIS, vmax=maxMIS, aspect='auto', interpolation='none',origin='lower', cmap=colormap)
    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(label_name,size=18)
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    
    return rho_inv, rr, misfit_inv, RMS, difff, diff



# +
import numpy as np
import matplotlib.pyplot as plt

def rho_inverse0_3C(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_lx, M_ly, M_lz,nr, begin,stop,test,a_s, n_it):

    # True density model
    m_true = 1/rho_true

    # Build data vector

    b      = np.zeros((N,nt_sub))
    K = c_input**2 *m_init  
    beta = 1/(K)

    for i in range(nt_sub):

        b[:,i] = (beta.flatten()*utt_g_st[:,:,i].flatten()) * (dist_r**2) *2
        
        
    ############################################################################################# 
    #  Inversion
    #############################################################################################
    A   = np.zeros((N, N,nt_sub))
    bbb = np.zeros((N,nt_sub))
    rr = np.zeros((N,nt_sub))

    # Damping
    I          = np.eye(N, N)
                 
    damp       = I*a_s**2
    
    Gx =  WW[:,:,np.newaxis]*(M_lx[:,:]) 
    Gy =  WW[:,:,np.newaxis]*(M_ly[:,:]) 
    Gz =  WW[:,:,np.newaxis]*(M_lz[:,:]) 
    GG=[]
    GG= np.append(Gx,Gy)
    GG= np.append(GG,Gz)
    GG = np.reshape(np.asarray(GG), (nr**2,nr**2,nt_sub))
    
    for n in range(0,nt_sub):
        
        G = GG[:,:,n]
        G_t = np.matrix.getH(G)

        A[:,:,n]     = np.dot(G_t,G) + damp  
        bbb[:,n]    = np.dot(G_t,b[:,n])+ (a_s**2 * np.dot(I,1/m_init.flatten())) 
    
    AA                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(AA), dd)#np.linalg.solve(M_l.sum(2),dd)
    rho_inv                   = np.reshape(1/rr, (nr,nr))    
        
        
        
    
    if test=='true':
    
        inv_LHS           = np.linalg.inv(A.sum(2))
        inv_RHS           = np.linalg.inv(bbb.sum(1)*np.eye(N,N))
        sum_inv_RHS       = np.diag(inv_RHS).sum(0)
        sum_sum_LHS       = np.diag(A.sum(2)).sum(0)

        m0                = 1/N * sum_inv_RHS * sum_sum_LHS
        dm                = np.dot(inv_LHS,(bbb.sum(1)*np.eye(N,N) -(A.sum(2)*(m0)) ))
        RHS               = bbb.sum(1) - (np.diag(A.sum(2))*(m0))
        p                 = np.reshape(np.diag(dm), (nr,nr))

        rr_test           = np.abs(np.diag(dm)+m0)
        
    
    
    
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.arange(0,len(m_true[1:-1,1:-1].flatten())),1/m_true[1:-1,1:-1].flatten(), label = 'True model', linewidth=2, color='tab:blue', marker='o',s=10)
    plt.plot(1/m_true[1:-1,1:-1].flatten(), linewidth=2, color='tab:blue')

    rrr = np.reshape(rr, (nr,nr))
    plt.scatter(np.arange(0,len(rrr[1:-1,1:-1].flatten())),1/rrr[1:-1,1:-1].flatten(), label='Inversion result',s=40,marker='+', color='coral')
    plt.plot(1/rrr[1:-1,1:-1].flatten(),color='coral')

    #plt.plot(1/rr.flatten(), label='Inversion result')
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{g}{cm^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    plt.show()
    
    m = m_true.flatten()
    r=rr.flatten()
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    #plt.legend()
    #plt.plot(m_init.flatten(), label='Initial guess')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    #plt.ylim(np.min(v),np.max(v))
    plt.show()
    
    
    rho_inv = np.reshape(1/rr, (nr,nr))
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap='pink_r')#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    #plt.colorbar()     
    plt.show() 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('True Density Model')
    plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap='pink_r') #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{g}{cm^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    #plt.colorbar()     
    plt.show() 
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    fig = plt.figure(figsize=(8,6))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] ,vmin=np.min((diff[1:-1,1:-1])), vmax=np.max((diff[1:-1,1:-1])), aspect='auto', interpolation='none',origin='lower', cmap=colormap)
    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(label_name,size=18)
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    #plt.colorbar()     
    #plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    
    return rho_inv, rr, misfit_inv, RMS, difff, diff

