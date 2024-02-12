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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)

def rho_inverse0(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it):
    
    
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
                 
    damp       = np.diag((a_s**2).flatten())

    for n in range(0,nt_sub):

        G =  WW*(M_l[:,:,n])
        G_t = np.matrix.getH(G)

        A[:,:,n]    = np.dot(G_t,G) + damp 
        bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 
    
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd) #np.linalg.solve(M_l.sum(2),dd)
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
    plt.legend(fontsize=14)
    plt.title(f'Inversion Output at Iteration $N^o$ {n_it}', fontsize=16)   
    plt.xlabel('Receiver station',fontsize=16)
    plt.ylabel(r'Density [$\frac{kg}{m^{3}}$]',fontsize=16)
    plt.xlim([0,len(rrr[1:-1,1:-1].flatten())])
    plt.show()
    
    m = m_true.T.flatten()
    r=rr.T.flatten()
    fig = plt.figure(figsize=(10,8))
    plt.plot(1/m[begin:stop], label = 'True model')
    plt.plot(1/r[begin:stop], label='Inversion result')
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('Model Parameter i')
    plt.ylabel('Density')
    plt.show()
    
    fig = plt.figure(figsize=(8,6))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    label_name =r'Density [$\frac{kg}{m^{3}}$]'
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)  
    plt.show() 
    
    fig = plt.figure(figsize=(8,6))
    plt.title('True Density Model')
    im_true=plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, 8.5, 0.5, 8.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{kg}{m^{3}}$]'
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label_name,size=18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    plt.show()
    
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    fig = plt.figure(figsize=(8,6))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] ,vmin=np.min((diff[1:-1,1:-1])), vmax=np.max((diff[1:-1,1:-1])), aspect='auto', interpolation='none',origin='lower', cmap=colormap,extent=[0.5, 8.5, 0.5, 8.5])
    fig.subplots_adjust(wspace=0.5) 
    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(label_name,size=18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)    
    plt.show() 

    print('RMS:',RMS)
   
    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)
    
    return rho_inv, rr, misfit_inv, RMS, difff, diff



# +
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colors = ["white", "wheat",'burlywood', 'tab:brown']
my_cmap = LinearSegmentedColormap.from_list("mycmap", colors)


def rho_inverse(rho_true, c_input, m_init, nt_sub, N, dist_r,utt_g_st,WW, M_l,nr, begin,stop,test,a_s, n_it,maxMIS, minMIS):

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

        A[:,:,n]    = np.dot(G_t,G) + damp 
        bbb[:,n]    = np.dot(G_t,b[:,n])+ ((a_s**2).flatten() * np.dot(I,1/m_init.flatten())) 
 
    GG                        = A.sum(2) 
    dd                        = bbb.sum(1)

    rr                        = np.dot(np.linalg.inv(GG), dd)
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
    plt.legend()
    plt.title('True model vs. Inversion Output')   
    plt.xlabel('model parameter i')
    plt.ylabel('rdensity')
    plt.show()

    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('Estimated Density Model - inversion output')
    im=plt.imshow(rho_inv[1:-1,1:-1] ,vmin=np.min(rho_inv[1:-1,1:-1]), vmax=np.max(rho_inv[1:-1,1:-1]), aspect='auto', interpolation='none',origin='lower', cmap=my_cmap,extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])#cmap='YlGnBu' ) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    label_name =r'Density [$\frac{kg}{m^{3}}$]'
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label=r'$[\frac{kg}{m^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    plt.show() 
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('True Density Model')
    im_true = plt.imshow(rho_true[1:-1,1:-1] ,vmin=np.min(rho_true[1:-1,1:-1]), vmax=np.max(rho_true[1:-1,1:-1]), aspect='auto', interpolation='none' ,origin='lower', cmap=my_cmap,extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5]) #,extent=[-3, 3, -3, 3]
    #plt.gca().invert_yaxis()
    plt.xlabel('X-axis',fontsize=16)
    plt.ylabel('Y-axis',fontsize=16)
    label_name =r'Density [$\frac{kg}{m^{3}}$]'
    cbar.set_label(label_name,size=18)
    cbar = plt.colorbar(im_true, format='%.0f')
    cbar.set_label(label=r'$[\frac{kg}{m^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    plt.show() 
    
    diff = 100*(rho_inv-rho_true)/rho_true
    difff = (rho_true-rho_inv)**2
    RMS  = np.sqrt((1/(nr-2)**2)*difff[1:-1,1:-1].sum(0).sum(0))
    
    fig,ax = plt.subplots(figsize=(10,8))
    plt.title('Parameter Error')
    label_name = '[%]'
    colormap='bwr'
    im=plt.imshow(diff[1:-1,1:-1] , aspect='auto', interpolation='none',origin='lower', cmap='bwr',extent=[0.5, rho_inv[1:-1,1].shape[0]+0.5, 0.5, rho_inv[1:-1,1].shape[0]+0.5])

    fig.subplots_adjust(wspace=0.5) #,extent=[-3, 3, -3, 3]
    #cax = divider.append_axes('right', size='5%', pad=0.2)
    #cbar.set_label(label_name,size=18)
#     plt.xlabel('X-axis',fontsize=16)
#     plt.ylabel('Y-axis',fontsize=16)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=16)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=16)
    cbar = plt.colorbar(im, format='%.0f')
    cbar.set_label(label=r'$[\frac{kg}{m^{3}}]$',size=24)
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    plt.xlabel(r'x receiver N$^{o}$',fontsize=20)
    plt.ylabel(r'y receiver N$^{o}$',fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    plt.show() 

    print('RMS:',RMS)
   

    #misfit_mref_data = 1/N *((1/m_true[1:-1,1:-1]-m_init[1:-1,1:-1])**2).sum(0).sum(0)
    misfit_inv = 1/N *((1/m_true[1:-1,1:-1]-rho_inv[1:-1,1:-1])).sum(0).sum(0)

    return rho_inv, rr, misfit_inv, RMS, difff, diff 

