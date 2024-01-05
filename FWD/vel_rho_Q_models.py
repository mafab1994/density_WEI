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

import matplotlib.pyplot as plt
import numpy as np
from salvus.mesh.simple_mesh import rho_from_gardeners, vs_from_poisson


def build_models_v_rho_lame(vel_model_type, rho_model_type, lame_model_type, vp0, vs0, rho0, lam0, mu0, xx, yy, zz):
    

        if lame_model_type == "homogeneous":
                    lam = np.ones_like(xx)*lam0
                    mu = np.ones_like(xx)*mu0
                    
                                

                    length = zz[0,0,:].shape[0]
                    rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 14 + (0.2)) * 200
                    rho[:,:,(int(length/6)+1)::] = np.ones_like(int(length/6)+1) * 2250
                    rho= rho[:,:,::-1]
 
                    vp = np.sqrt( (lam+(2*mu)) / rho)
                    vs = np.sqrt(mu/rho)
                    
        return vp, vs, rho, lam, mu


# +
def build_models_v_rho(vel_model_type, rho_model_type, vp0, vs0, rho0, xx, yy, zz):
    
    if (vel_model_type == rho_model_type) == True:
        model_type = vel_model_type

        if model_type == "homogeneous":

                    vp = np.ones_like(xx)*vp0
                    vs = np.ones_like(xx)*vs0
                    rho = rho_from_gardeners(vp)


        elif model_type == "linear_grad":

                    vp = np.ones_like(xx)
                    vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                    
                    for i in range(0,len(vec_vp)):
                        vp[:,:,i] =  vp[:,:,i]*vec_vp[i]
                    #vp[:,:,0:20] = vp0
                    #vp[:,:,20:40] = vp0+100
                    #vp[:,:,40:60] = vp0+200
                    #vp[:,:,60:80] = vp0+300
                    #vp[:,:,80:110] = vp0+400
                    #vp[:,:,110:zz[0,0,:].shape[0]] = vp0+500
                    

                    vs = vs_from_poisson(vp)
                    vp = vp[:,:,::-1]
                    vs = vs[:,:,::-1]

                    rho = rho_from_gardeners(vp)
                    
        elif model_type == "linear_gradB":

                    vp = np.ones_like(xx)
                    vec_vp =  np.linspace(vp0, vp0+500, zz[0,0,:].shape[0] )

                    for i in range(0,len(vec_vp)):
                        vp[:,:,i] =  vp[:,:,i]*vec_vp[i]

                    vs = vs_from_poisson(vp)
                    vp = vp[:,:,::-1]
                    vs = vs[:,:,::-1]

                    rho = rho_from_gardeners(vp)
                    
        elif model_type == "het_z":

                    vp = np.ones_like(xx)
                    length = zz[0,0,:].shape[0]


#                     vp[:,0:(int(length/6)+1),0:(int(length/6)+1)] =  vp0
#                     vp[:,(int(length/6)+1)::,0:(int(length/6)+1)] =  vp0 + 300
#                     vp[:,:,(int(length/6)+1)::] =  vp0 + 1000#20 #200 test2
                    vp[:,:,0:(int(length/6)+1)] =  vp0
                    vp[:,:,(int(length/6)+1)::] =  vp0 + 1000#20 #200 test2

                    vs = vs_from_poisson(vp)
                    vp = vp[:,:,::-1]
                    vs = vs[:,:,::-1]

                    rho = rho_from_gardeners(vp)
                    
        elif model_type == "het_z_lithgoe":

                    vp = np.ones_like(xx)
                    length = zz[0,0,:].shape[0]

                    vp[:,:,0:(int(length/5)+1)] =  vp0
                    vp[:,:,(int(length/5)+1)::] =  vp0 + 3300

                    vs = vs_from_poisson(vp)
                    vp = vp[:,:,::-1]
                    vs = vs[:,:,::-1]

                    rho = rho_from_gardeners(vp)
                    
        elif model_type == "het_z+FAULT":

                    vp = np.ones_like(xx)
                    length = zz[0,0,:].shape[0]
                    len_x = xx[0,:,0].shape[0]


#                     vp[:,0:(int(length/6)+1),0:(int(length/6)+1)] =  vp0
#                     vp[:,(int(length/6)+1)::,0:(int(length/6)+1)] =  vp0 + 300
#                     vp[:,:,(int(length/6)+1)::] =  vp0 + 1000#20 #200 test2
                    vp[:,:,0:(int(length/6)+1)] =  vp0 + 1000
                    vp[:,:,(int(length/6)+1)::] =  vp0 + 2000#20 #200 test2
        
                    #fault zone 
                    zzz = 0
                    for ii in range(275,len_x-275):
                        #for zzz in range(0,(int(length/6)+1)):
                            vp[ii-4:ii+4,275:281,zzz] =  vp0 
                            #vp[ii-2:ii+2,ii-2:ii+2,zzz+1] =  vp0 / 4
                            zzz = zzz+1

                
                    vs = vs_from_poisson(vp)
                    vp = vp[:,:,::-1]
                    vs = vs[:,:,::-1]

                    rho = rho_from_gardeners(vp)
                    
        
        elif model_type == "sin_dist":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                vp = 1500.0 * np.ones_like(xx) - yy * 0.1 + np.sin(xx / 8 + 0.5) * 2
                vs = 1000.0 * np.ones_like(xx) - yy * 0.1 + np.sin(xx / 12 - 0.3) * 2
                rho = 980.0 * np.ones_like(xx) - yy * 0.1 + np.sin(xx / 15 + 1) * 2
                
        elif model_type == "heterogeneous_xy":
                vp = np.ones_like(xx)
                vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

                for i in range(0,len(vec_vp)):
                        vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

                vs = vs_from_poisson(vp)
                vp = vp[:,:,:]
                vs = vs[:,:,:]

                rho = rho_from_gardeners(vp)
                
        elif model_type == "heterogeneous_xy_hetZ":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]

                length = zz[0,0,:].shape[0]
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 200
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]
                rho = rho_from_gardeners(vp)
            


                
        elif model_type == "heterogeneous_xy_hetZ_II":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]
                length = zz[0,0,:].shape[0]
                #vp = 1850.0 * np.ones_like(xx)-np.sin(xx / 18 + (0.2)) * 1000
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 25 + (0.2)) * 300
        
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1800#20 #200 test2
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]
                rho = rho_from_gardeners(vp)
         
    else:

        if vel_model_type == "homogeneous":
                vp = np.ones_like(xx)*vp0
                vs = np.ones_like(xx)*vs0
                
        elif vel_model_type == "linear_xy":
                vp = np.ones_like(xx)
                vec_vp =  np.linspace(vp0, vp0+1500, xx[:,0,0].shape[0] )

                for i in range(0,len(vec_vp)):
                        vp[i,:,:] =  vp[i,:,:]*vec_vp[i]
                length = zz[0,0,:].shape[0]
                vp[:,:,(int(length/6)+1)::] =  vp0 + 2500

                vs = vs_from_poisson(vp)
                #vp = vp[:,:,::-1]
                #vs = vs[:,:,::-1]
                
        elif vel_model_type == "het_z":

                    vp = np.ones_like(xx)
                    length = zz[0,0,:].shape[0]

                    vp[:,:,0:(int(length/6)+1)] =  vp0
                    vp[:,:,(int(length/6)+1)::] =  vp0 + 1000#20 #200 test2

                    vs = vs_from_poisson(vp)
                    vp = vp[:,:,::-1]
                    vs = vs[:,:,::-1]


        elif vel_model_type == "linear_grad":
                vp = np.ones_like(xx)
                vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )

                for i in range(0,len(vec_vp)):
                        vp[:,:,i] =  vp[:,:,i]*vec_vp[i]
                        
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]
                
        elif vel_model_type == "heterogeneous_xy":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]

                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 16 + (0.2)) * 600
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                vs = vs_from_poisson(vp)
                
        elif vel_model_type == "heterogeneous_xy_hetZ":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]
                length = zz[0,0,:].shape[0]
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 300
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]

        elif vel_model_type == "heterogeneous_xy_hetZ2":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]
                length = zz[0,0,:].shape[0]
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 40 + (0.2)) * 300
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]
                
        elif vel_model_type == "heterogeneous_xy_hetZ_TEST":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]
                length = zz[0,0,:].shape[0]
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 50
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]

        elif vel_model_type == "heterogeneous_xy_hetZ_TEST":

                length = zz[0,0,:].shape[0]
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 14 + (0.2)) * 300
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]
                
        elif vel_model_type == "heterogeneous_xy_hetZ_DIAG":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]
                length = zz[0,0,:].shape[0]
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 300
                vp_diag = 1850.0 * np.ones_like(xx)-np.cos(yy / 18 + (0.2)) * 300
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp_diag[:,:,(int(length/6)+1)::] + 1500
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]
                
        elif vel_model_type == "heterogeneous_xy_hetZ_contrast2":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]
                length = zz[0,0,:].shape[0]
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 6 + (0.2)) * 300
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]
                
        elif vel_model_type == "heterogeneous_xy_hetZ_II":
#                 vp = np.ones_like(xx)
#                 vec_vp =  np.linspace(vp0, vp0+3000, zz[0,:,0].shape[0] )

#                 for i in range(0,len(vec_vp)):
#                         vp[:,i,:] =  vp[:,i,:]*vec_vp[i]

#                 vs = vs_from_poisson(vp)
#                 vp = vp[:,:,:]
#                 vs = vs[:,:,:]
                length = zz[0,0,:].shape[0]
                vp = 1850.0 * np.ones_like(xx)-np.sin(yy / 75 + (0.2)) * 1000
                #vp = vp.T
                vp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                vs = vs_from_poisson(vp)
                vp = vp[:,:,::-1]
                vs = vs[:,:,::-1]
        
                
        if rho_model_type =="homogeneous":
                rho = np.ones_like(xx)*rho0
                
        elif rho_model_type == "het_z":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                length = zz[0,0,:].shape[0]
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = np.mean(1600.0 * np.ones_like(yy)-np.sin(xx / 14 + (0.2)) * 400) *np.ones_like(yy)
                vpp = 1850.0 * np.ones_like(xx)-np.sin(xx / 18 + (0.2)) * 300
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
        elif rho_model_type == "linear_yx":
                rho = np.ones_like(yy)
                vec_rho =  np.linspace(rho0, rho0+2500, yy[0,:,0].shape[0] )

                for i in range(0,len(vec_rho)):
                        rho[:,i,:] =  rho[:,i,:]*vec_rho[i]
                length = zz[0,0,:].shape[0]
                rho[:,:,(int(length/6)+1)::] =  rho0 + 3500
                #rho= rho[:,:,::-1]
                

                
        elif rho_model_type =="heterogeneous_xy":
                #rho = 980.0 * np.ones_like(xx) - yy * 0.1 + np.sin(xx / 15 + 1) * 2
                rho = np.ones_like(xx)
                length = yy[0,:,0].shape[0]

                rho[:,0:(int(length/2)+1),:] =  rho0
                rho[:,(int(length/2)+1)::,:] =  rho0 + 1000#20 #200 test2
                
        elif rho_model_type == "linear_grad":
                rho = np.ones_like(xx)
                rho_vec =  np.linspace(rho0, rho0+1000, zz[0,0,:].shape[0] )
                
                for i in range(0,len(rho_vec)):
                        rho[:,:,i] =  rho[:,:,i]*rho_vec[i]

                rho = rho[:,:,::-1]
                
        elif rho_model_type == "sin_dist":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                #rho = 980.0 * np.ones_like(xx) - yy * 1.9 + np.sin(xx / 15 + 1) * 2
                
                x = np.ones(xx[:,0,0].shape[0])*1400
                y = np.ones(yy[0,:,0].shape[0])*1400
                z = np.ones(zz[0,0,:].shape[0])*1400
                rxx, ryy,rzz = np.meshgrid(x, y, z, indexing="ij")
                
                x_ins = np.linspace(500, 2500, 40)
                y_ins = np.linspace(500, 2500, 40)
                z_ins = np.ones((300))*1400
                
                rxx_ins, ryy_ins,rzz_ins = np.meshgrid(x_ins, y_ins, z_ins, indexing="ij")
                
                rxx[285:325,285:325,0:300] = rxx_ins
                
                rho = rxx
                
        elif rho_model_type == "sin_dist2":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 8 + (0.2)) * 600
                
        elif rho_model_type == "sin_dist2b":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 14 + (0.2)) * 400
                
        elif rho_model_type == "test":
            
                length = zz[0,0,:].shape[0]
                rho = np.ones_like(xx)*rho0
                rho[300:338,300:338,:] = 200+300
                vpp = (1850.0+300) * np.ones_like(xx)
                vpp[:,:,(int(length/6)+1)::] =  vp0 + (1500+300)#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
            
        elif rho_model_type == "sin_dist2b_hetZ":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                length = zz[0,0,:].shape[0]
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 14 + (0.2)) * 400
                vpp = 1850.0 * np.ones_like(xx)-np.sin(xx / 18 + (0.2)) * 300
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
                
        elif rho_model_type == "sin_dist2b_hetZ2":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                length = zz[0,0,:].shape[0]
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 80 + (0.2)) * 400
                vpp = 1850.0 * np.ones_like(xx)-np.sin(xx / 40 + (0.2)) * 300
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
        elif rho_model_type == "sameDIR":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                length = zz[0,0,:].shape[0]
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(yy / 40 + (0.2)) * 400
                vpp = 1850.0 * np.ones_like(xx)-np.sin(yy / 20 + (0.2)) * 300
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
        elif rho_model_type == "sameDIR2":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                length = zz[0,0,:].shape[0]
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(yy / 30 + (0.2)) * 400
                vpp = 1850.0 * np.ones_like(xx)-np.sin(yy / 20 + (0.2)) * 300
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
        elif rho_model_type == "SAME":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                length = zz[0,0,:].shape[0]
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 300
                vpp = 1850.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 200
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
        elif rho_model_type == "SAME_STRONGER":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                length = zz[0,0,:].shape[0]
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 700
                vpp = 1850.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 200
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                

        elif rho_model_type == "sin_dist2b_hetZ_TEST":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 14 + (0.2)) * 400
                vpp = 1850.0 * np.ones_like(xx)-np.sin(xx / 18 + (0.2)) * 50
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
    
        elif rho_model_type == "XY_sin_dist2b_hetZ":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 14 + (0.2)) * 300  + np.sin(yy / 20 + (0.2)) * 100
                vpp = 1850.0 * np.ones_like(xx)-np.sin(xx / 18 + (0.2)) * 300
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
                
        elif rho_model_type == "sin_dist2b_hetZ_DIAG":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 14 + (0.2)) * 400
                #vpp = 1850.0 * np.ones_like(xx)-np.sin(xx / 18 + (0.2)) * 300
                #vp_diag = 1850.0 * np.ones_like(xx)-np.sin(yy / 18 + (0.2)) * 300 #+np.cos(yy / 18 + (0.2)) * 300
                vp_diag = (1850.0 * np.ones_like(xx**2) - yy * 2 + np.sin(xx / 19 +0.2) + ((xx/0.4)))# +1500
                #vp = vp.T
                #vpp[:,:,(int(length/6)+1)::] =  (1850.0 * np.ones_like(xx)-np.cos(xx / 18 + (0.2)) * 300) + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vp_diag[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
                
        elif rho_model_type == "sin_dist2b_hetZ_II":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 45 + (0.2)) * 900
                vpp = 1850.0 * np.ones_like(xx)-np.sin(xx / 75 + (0.2)) * 1000
                #vp = vp.T
                vpp[:,:,(int(length/6)+1)::] =  vp0 + 1500#20 #200 test2
                rho[:,:,(int(length/6)+1)::] = rho_from_gardeners(vpp[:,:,(int(length/6)+1)::])
                rho= rho[:,:,::-1]
                
        elif rho_model_type == "sin_dist3":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = (1600.0 * np.ones_like(yy)-np.sin(xx / 8 + (0.2)) * 600) + 1000
                rho2 = 1600.0 * np.ones_like(xx)-np.sin(yy / 4 + (0.01)) * 200 +np.cos(xx / 6 + (0.01)) * 200
                length = zz[0,0,:].shape[0]
                rho[:,:,0:(int(length/6)+1)] =  rho2[:,::-1,0:(int(length/6)+1)] - 500
                rho = rho[:,:,::-1]
                
        elif rho_model_type == "sin_dist4":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = (1600.0 * np.ones_like(yy)-np.sin(xx / 8 + (0.2)) * 600) + 1000
                rho2 = 1600.0 * np.ones_like(xx)-np.sin(yy / 4 + (0.01)) * 200 +np.cos(xx / 6 + (0.01)) * 200
                length = zz[0,0,:].shape[0]
                rho[:,:,0:(int(length/6)+1)] =  rho2[:,:,0:(int(length/6)+1)] - 500
                rho = rho[:,:,::-1]
                
                
        elif rho_model_type == "sin_dist2_zstep":
                # Simplistic 5 parameter model with a depth gradient and
                # some sinusoidal perturbation.
                # This is just an example but this would be the place to add
                # your own velocity model.
                #vec_vp =  np.linspace(vp0, vp0+1000, zz[0,0,:].shape[0] )
                
                #rho = 2000.0 * np.ones_like(yy) - xx * 2.5 + np.sin(xx / 8 + 0.5) * 200#+ np.sin(xx / 15 + 1) * 5
                rho = 1600.0 * np.ones_like(yy)-np.sin(xx / 8 + (0.2)) * 600
                rho_max= np.max(rho)
                length = zz[0,0,:].shape[0]
                rho[:,:,(int(length/6)+1)::] =  rho_max + 1000
                rho = rho[:,:,::-1]
                
    return vp, vs, rho


# -

def build_models_Q(Q_model, xx,yy):
    
    if Q_model == "homogeneous-":
        qkappa = 400 * np.ones_like(xx) # 40 * np.ones_like(xx) #2000.0 * np.ones_like(xx) 
        qmu = 200 * np.ones_like(xx) #75.0 * np.ones_like(xx) 
        
    if Q_model == "homogeneous":
        qkappa = 40 * np.ones_like(xx) # 40 * np.ones_like(xx) #2000.0 * np.ones_like(xx) 
        qmu = 20 * np.ones_like(xx) #75.0 * np.ones_like(xx) 
   
    if Q_model == "homogeneous+":
        qkappa = 15 * np.ones_like(xx) #2000.0 * np.ones_like(xx) 
        qmu = 7 * np.ones_like(xx) #75.0 * np.ones_like(xx) 
        
    if Q_model == "homogeneous++":
        qkappa = 5 * np.ones_like(xx) #2000.0 * np.ones_like(xx) 
        qmu = 2 * np.ones_like(xx) #75.0 * np.ones_like(xx) 
    
    elif Q_model == 'sin_dist':

        qkappa = 1850.0 * np.ones_like(xx)-np.sin(yy / 9 - (0.2)) * 300 #2000.0 * np.ones_like(xx) - yy * 2.8 
        qmu = 475.0 * np.ones_like(xx)-np.sin(yy / 9 - (0.2)) * 300         #75.0 * np.ones_like(xx) - yy * 0.1 
        #qkappa = 2000.0 * np.ones_like(xx) - yy * 0.1 + np.sin(xx / 6) * 2
        #qmu = 75.0 * np.ones_like(xx) - yy * 0.1 + np.sin(xx / 10 - 1.2) * 2
    elif Q_model == 'circular':
        
        qkappa = 1850.0 * np.ones_like(xx)- (np.sin(yy / 12 + (0.2)) * 100) + (np.cos(xx / 14 - (0.4))*100)  #2000.0 * np.ones_like(xx) - yy * 2.8 
        qmu = 250.0 * np.ones_like(xx) - (np.sin(yy / 12 + (0.2)) * 40) + (np.cos(xx / 14 - (0.4))*20 )      #75.0 * np.ones_like(xx) - yy * 0.1
   
    elif Q_model == 'circular_strong':
        
        qkappa = 1850.0 * np.ones_like(xx)- (np.sin(yy / 12 + (0.2)) * 40) + (np.cos(xx / 14 - (0.4))*40) -1300  #2000.0 * np.ones_like(xx) - yy * 2.8 
        qmu = 250.0 * np.ones_like(xx) - (np.sin(yy / 12 + (0.2)) * 20) + (np.cos(xx / 14 - (0.4))*20 )      #75.0 * np.ones_like(xx) - yy * 0.1
   
    elif Q_model == 'circular_stronger':
        
        qkappa = (1850.0 * np.ones_like(xx)- (np.sin(yy / 12 + (0.2)) * 40) + (np.cos(xx / 14 - (0.4))*40) -1300)/10 -10  #2000.0 * np.ones_like(xx) - yy * 2.8 
        qmu = 250.0 * np.ones_like(xx) - (np.sin(yy / 12 + (0.2)) * 20) + (np.cos(xx / 14 - (0.4))*20 )      #75.0 * np.ones_like(xx) - yy * 0.1
    
    elif Q_model == 'circular_stronger+':
        
        qkappa = ((1850.0 * np.ones_like(xx)- (np.sin(yy / 12 + (0.2)) * 40) + (np.cos(xx / 14 - (0.4))*40) -1300)/10 )-45 #2000.0 * np.ones_like(xx) - yy * 2.8 
        qmu = 250.0 * np.ones_like(xx) - (np.sin(yy / 12 + (0.2)) * 20) + (np.cos(xx / 14 - (0.4))*20 )      #75.0 * np.ones_like(xx) - yy * 0.1
    return qmu, qkappa
