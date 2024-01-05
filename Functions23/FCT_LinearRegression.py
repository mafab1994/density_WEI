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

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def LinReg_c3D(acc_internal,dttv_internal, nt_sub, ll, N_internal, nrx, nry, nrz,order):
      
    acc_internal = np.reshape(acc_internal,(N_internal, nt_sub,ll))
    dttv_internal = np.reshape(dttv_internal,(N_internal, nt_sub,ll))

    model = LinearRegression()
    m    = np.zeros((nrx, nry, nrz))
    r_sq = np.zeros((nrx, nry, nrz))

    c_phase = np.zeros((N_internal, ll))
    m = np.zeros((N_internal, ll))
    r_sq = np.zeros((N_internal, ll))
    for fi in range(ll):
        for i in range(N_internal):
            model.fit(acc_internal[i,:,fi].reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            r_sq[i,fi]  = model.score((acc_internal[i,:,fi]).reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            inter       = model.intercept_
            m[i,fi]     = model.coef_
            c_phase[i,fi] = np.sqrt(np.abs(m[i,fi]))
    
    c_phase = np.reshape(c_phase, (nrx,nry,nrz,ll))
    m = np.reshape(m, (nrx,nry,nrz,ll))
    r_sq = np.reshape(r_sq, (nrx,nry,nrz,ll))
    
    return c_phase, m, r_sq


def LinReg_c(acc_internal,dttv_internal, nt_sub, ll, N_internal, nrx, nry, nrz,order):
      
    acc_internal = np.reshape(acc_internal,(N_internal, nt_sub,ll))
    dttv_internal = np.reshape(dttv_internal,(N_internal, nt_sub,ll))

    model = LinearRegression()
    m    = np.zeros((nrx-order, nry-order, nrz-order))
    r_sq = np.zeros((nrx-order, nry-order, nrz-order))

    c_phase = np.zeros((N_internal, ll))
    m = np.zeros((N_internal, ll))
    r_sq = np.zeros((N_internal, ll))
    for fi in range(ll):
        for i in range(N_internal):
            model.fit(acc_internal[i,:,fi].reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            r_sq[i,fi]  = model.score((acc_internal[i,:,fi]).reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            inter       = model.intercept_
            m[i,fi]     = model.coef_
            c_phase[i,fi] = np.sqrt(np.abs(m[i,fi]))
    
    c_phase = np.reshape(c_phase, (nrx-order,nry-order,nrz-order,ll))
    m = np.reshape(m, (nrx-order,nry-order,nrz-order,ll))
    r_sq = np.reshape(r_sq, (nrx-order,nry-order,nrz-order,ll))
    
    return c_phase, m, r_sq


def LinReg_rho(acc_internal,dttv_internal, nt_sub, ll, N_internal, nrx, nry, nrz,order):
      
    acc_internal = np.reshape(acc_internal,(N_internal, nt_sub,ll))
    dttv_internal = np.reshape(dttv_internal,(N_internal, nt_sub,ll))

    model = LinearRegression()
    m    = np.zeros((nrx-order, nry-order, nrz-order))
    r_sq = np.zeros((nrx-order, nry-order, nrz-order))

    rho_ratio = np.zeros((N_internal, ll))
    m = np.zeros((N_internal, ll))
    r_sq = np.zeros((N_internal, ll))
    for fi in range(ll):
        for i in range(N_internal):
            model.fit(acc_internal[i,:,fi].reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            r_sq[i,fi]  = model.score((acc_internal[i,:,fi]).reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            inter       = model.intercept_
            m[i,fi]     = model.coef_
            rho_ratio[i,fi] = m[i,fi]
    
    rho_ratio = np.reshape(rho_ratio, (nrx-order,nry-order,nrz-order,ll))
    m = np.reshape(m, (nrx-order,nry-order,nrz-order,ll))
    r_sq = np.reshape(r_sq, (nrx-order,nry-order,nrz-order,ll))
    
    return rho_ratio, m, r_sq


def LinReg_rho_surf(acc_internal,dttv_internal, nt_sub, ll, N_internal, nrx, nry,order):
      
    acc_internal = np.reshape(acc_internal,(N_internal, nt_sub,ll))
    dttv_internal = np.reshape(dttv_internal,(N_internal, nt_sub,ll))

    model = LinearRegression()
    m    = np.zeros((nrx-order, nry-order))
    r_sq = np.zeros((nrx-order, nry-order))

    rho_ratio = np.zeros((N_internal, ll))
    m = np.zeros((N_internal, ll))
    r_sq = np.zeros((N_internal, ll))
    for fi in range(ll):
        for i in range(N_internal):
            model.fit(acc_internal[i,:,fi].reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            r_sq[i,fi]  = model.score((acc_internal[i,:,fi]).reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            inter       = model.intercept_
            m[i,fi]     = model.coef_
            rho_ratio[i,fi] = m[i,fi]
    
    rho_ratio = np.reshape(rho_ratio, (nrx-order,nry-order,ll))
    m = np.reshape(m, (nrx-order,nry-order,ll))
    r_sq = np.reshape(r_sq, (nrx-order,nry-order,ll))
    
    return rho_ratio, m, r_sq


def LinReg_c2(acc_internal,dttv_internal, nt_sub, ll, N_internal, nrx, nry, nrz,order):
      
    acc_internal = np.reshape(acc_internal,(N_internal, nt_sub,ll))
    dttv_internal = np.reshape(dttv_internal,(N_internal, nt_sub,ll))

    model = LinearRegression()
    m    = np.zeros((nrx-order, nry-order, nrz-order))
    r_sq = np.zeros((nrx-order, nry-order, nrz-order))

    c_phase = np.zeros((N_internal, ll))
    m = np.zeros((N_internal, ll))
    r_sq = np.zeros((N_internal, ll))
    for fi in range(ll):
        for i in range(N_internal):
            model.fit(acc_internal[i,:,fi].reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            r_sq[i,fi]  = model.score((acc_internal[i,:,fi]).reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            inter       = model.intercept_
            m[i,fi]     = model.coef_
            c_phase[i,fi] = np.sqrt(np.abs(m[i,fi]))
    
    c_phase = np.reshape(c_phase, (nrx-order,nry-order,ll))
    m = np.reshape(m, (nrx-order,nry-order,ll))
    r_sq = np.reshape(r_sq, (nrx-order,nry-order,ll))
    
    return c_phase, m, r_sq


def LinReg_c2_2DACC(acc_internal,dttv_internal, nt_sub, ll, N_internal, nrx, nry, order):
      
    acc_internal = np.reshape(acc_internal,(N_internal, nt_sub,ll))
    dttv_internal = np.reshape(dttv_internal,(N_internal, nt_sub,ll))

    model = LinearRegression()
    m    = np.zeros((nrx, nry))
    r_sq = np.zeros((nrx, nry))

    c_phase = np.zeros((N_internal, ll))
    m = np.zeros((N_internal, ll))
    r_sq = np.zeros((N_internal, ll))
    for fi in range(ll):
        for i in range(N_internal):
            model.fit(acc_internal[i,:,fi].reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            r_sq[i,fi]  = model.score((acc_internal[i,:,fi]).reshape(-1, 1), dttv_internal[i,:,fi].reshape(-1, 1))
            inter       = model.intercept_
            m[i,fi]     = model.coef_
            c_phase[i,fi] = np.sqrt(np.abs(m[i,fi]))
    
    c_phase = np.reshape(c_phase, (nrx,nry,ll))
    m = np.reshape(m, (nrx,nry,ll))
    r_sq = np.reshape(r_sq, (nrx,nry,ll))
    
    return c_phase, m, r_sq


def LinReg_PLOT(acc_internal,dttv_internal, nt_sub, ll, N_internal, nrx, nry, nrz, posX, posY, posZ, nf, m,r_sq, f_filt_cent, filtered):
    plt.figure(figsize=(10,8))
    plt.scatter(acc_internal[posX,posY,posZ,:,nf].reshape(-1, 1), dttv_internal[posX,posY,posZ,:,nf].reshape(-1, 1))
    plt.plot(acc_internal[posX,posY,posZ,:,nf].reshape(-1, 1), m[posX,posY,posZ,nf] * acc_internal[posX,posY,posZ,:,nf].reshape(-1, 1), c='red', linewidth=0.5)
    plt.xlabel(r'$[\partial_{t}^{2}u(x,y,t)]$', fontsize=14)
    plt.ylabel(r'$[\nabla^{2}u(x,y,t)]$', fontsize=14)
    plt.title(f"Receiver [{posX}, {posY}, {posZ}] ; CD = {np.round(r_sq[posX,posY,posZ,nf],4)}", fontsize=14)
    if filtered=='no':
        plt.suptitle('Unfiltered')
    elif filtered=='yes':
        plt.suptitle('Filtered at ' +str(f_filt_cent[nf]) +' Hz')
    
    plt.show()
