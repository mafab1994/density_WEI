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


# +
c_min = 100                  #our first estimate of phase medium velocity
f = np.linspace(1,20,200)    #frequencies between 1&20
lam = c_min/f                #wavelength
dx_list = [1,2,4,8,12,16,20] #possible spacings

err = np.zeros((len(dx_list),len(f)))

i=0
for i in range(0,len(dx_list)):
    err[i,:] = 100 * ( (dx_list[i]**2) * (2 * np.pi)**2 ) / (24 * (lam)**2)
    
# -

fraction = np.round(1/(dx_list[2]/lam),1) # spacing criterion for dx=4m

args=np.argwhere(err[2,:]>10) #get indice of where error is larger than 10%
#args[0]-1 will be the indice closest at intersection between finite difference error curve and error threshold of 10% (line: y=10)

# +
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(f,err[2,:], color='blue', label=r'dx=4')
plt.ylabel('Finite Difference Error [%]', fontsize=14)
plt.xlabel('Frequencies [Hz]', fontsize=14)
plt.plot(f,np.ones((len(f)))*10, linestyle=':',color='black')
plt.scatter(f[args[0]-1],err[2,args[0]-1], marker='*', s=400, color='gold')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.65,'Minimum medium velocity: \n'+str(c_min)+' [m/s] ', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.text(0.05, 0.5,'Threshold frequency: \n'+str(np.round(f[args[0]-1][0],1))+' [Hz] ', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.text(0.05, 0.35,'Sampling criterion: \n'+r' $\lambda$/'+str(np.round(fraction[args[0]-1][0],0)), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.xlim([f[0],f[-1]])
ax.legend(fontsize=14,loc='upper left', ncol=1)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)

plt.show()
# -


