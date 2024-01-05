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


def NORM_SOL_SYSTEM_3D_ELASTIC(dttv1_internal, dttv2_internal, dttv3_internal, rotROT1_internal, rotROT2_internal, rotROT3_internal,\
                         gradDIV1_internal, gradDIV2_internal, gradDIV3_internal, nt_range, nrx, nry, nrz, order,nn):
    
    N_internal        = dttv3_internal[:,:,:,0,0].flatten().shape[0]
    dttv3_int    = np.reshape(dttv3_internal, (N_internal,nt_range, len(nn)))
    dttv2_int    = np.reshape(dttv2_internal, (N_internal,nt_range, len(nn)))
    dttv1_int    = np.reshape(dttv1_internal, (N_internal,nt_range, len(nn)))
    gradDIV3_int = np.reshape(gradDIV3_internal, (N_internal,nt_range, len(nn)))
    rotROT3_int  = np.reshape(rotROT3_internal, (N_internal,nt_range, len(nn)))
    gradDIV2_int = np.reshape(gradDIV2_internal, (N_internal,nt_range, len(nn)))
    rotROT2_int = np.reshape(rotROT2_internal, (N_internal,nt_range, len(nn)))
    gradDIV1_int = np.reshape(gradDIV1_internal, (N_internal,nt_range, len(nn)))
    rotROT1_int  = np.reshape(rotROT1_internal, (N_internal,nt_range, len(nn)))


    ## Normal Solution from vertical particle velocity - 3D case
    body_est = np.zeros((N_internal,3,len(nn)))
    for fi in range(len(nn)):
        for i in range(N_internal):
            #for ttt in range(1000,nt_sub):
                        #X = np.array([[gradDIV1_int[i,:],-rotROT1_int[i,:]],[gradDIV2_int[i,:],-rotROT2_int[i,:]],\
                        #             [gradDIV3_int[i,:],-rotROT3_int[i,:]]])
                        Xl = []
                        Xl = np.append(gradDIV1_int[i,:,fi], gradDIV2_int[i,:,fi])
                        Xl = np.append(Xl, gradDIV3_int[i,:,fi])


                        Xr = []
                        Xr = np.append(-rotROT1_int[i,:,fi], -rotROT2_int[i,:,fi])
                        Xr = np.append(Xr, -rotROT3_int[i,:,fi])

                        X = np.array([np.ones((3*nt_range)),Xl, Xr]).T
                        #X= X.flatten()
                        #X = np.reshape(X, (3,nt_range,2))
                        #X = np.reshape(X,(3*nt_range,2)) #+ 1e-30
                        #y_rhs = np.array([dttv1_int[i,:],dttv2_int[i,:],dttv3_int[i,:]])
                        #y_rhs = y_rhs.flatten()
                        #y_rhs = np.reshape(y_rhs,(3*nt_range))
                        y_rh = []
                        y_rh=np.append(dttv1_int[i,:,fi],dttv2_int[i,:,fi])
                        y_rhs=np.append(y_rh,dttv3_int[i,:,fi])


                        inv = np.linalg.inv(np.dot(X.T,X))
                        XX = np.dot(inv, X.T)
                        SOL = np.dot(XX, y_rhs)
                        body_est[i,:,fi] = np.sqrt(abs(SOL))
    if order ==2:
        vs_est_3eq = np.reshape(body_est[:,2], (nrx-2,nry-2,nrz-2, len(nn)))
        vp_est_3eq = np.reshape(body_est[:,1], (nrx-2,nry-2,nrz-2, len(nn)))
        sc_term = np.reshape(body_est[:,0], (nrx-2,nry-2,nrz-2, len(nn)))
    if order ==4:
        vs_est_3eq = np.reshape(body_est[:,2], (nrx-4,nry-4,nrz-4, len(nn)))
        vp_est_3eq = np.reshape(body_est[:,1], (nrx-4,nry-4,nrz-4, len(nn)))
        sc_term = np.reshape(body_est[:,0], (nrx-4,nry-4,nrz-4, len(nn)))


    return vp_est_3eq ,vs_est_3eq,  sc_term


def NORM_SOL_SYSTEM_INHOM_CENTRAL_FD_3D_ELASTIC(dttv1_internal, dttv2_internal, dttv3_internal, rotROT1_internal, rotROT2_internal, rotROT3_internal,\
                           gradDIV1_internal, gradDIV2_internal, gradDIV3_internal, nt_range, nrx, nry, nrz, order,\
                                 divU_FD_dx_internal, divU_FD_dy_internal, divU_FD_dz_internal,\
                          Gradgrad_dx3_internal, Gradgrad_dy3_internal, Gradgrad_dz3_internal,nn, dx,dy,dz):
    
    N_internal        = dttv3_internal[:,:,:,0,0].flatten().shape[0]
    
    dttv3_int    = np.reshape(dttv3_internal, (N_internal,nt_range, len(nn)))
    dttv2_int    = np.reshape(dttv2_internal, (N_internal,nt_range, len(nn)))
    dttv1_int    = np.reshape(dttv1_internal, (N_internal,nt_range, len(nn)))
    gradDIV3_int = np.reshape(gradDIV3_internal, (N_internal,nt_range, len(nn)))
    rotROT3_int  = np.reshape(rotROT3_internal, (N_internal,nt_range, len(nn)))
    gradDIV2_int = np.reshape(gradDIV2_internal, (N_internal,nt_range, len(nn)))
    rotROT2_int = np.reshape(rotROT2_internal, (N_internal,nt_range, len(nn)))
    gradDIV1_int = np.reshape(gradDIV1_internal, (N_internal,nt_range, len(nn)))
    rotROT1_int  = np.reshape(rotROT1_internal, (N_internal,nt_range, len(nn)))
    
    divU_FD_dx_int = np.reshape(divU_FD_dx_internal, (N_internal,nt_range, len(nn)))
    divU_FD_dy_int = np.reshape(divU_FD_dy_internal, (N_internal,nt_range, len(nn)))
    divU_FD_dz_int = np.reshape(divU_FD_dz_internal, (N_internal,nt_range, len(nn)))
    Gradgrad_dx3_int = np.reshape(Gradgrad_dx3_internal, (N_internal,nt_range, len(nn)))
    Gradgrad_dy3_int  = np.reshape(Gradgrad_dy3_internal, (N_internal,nt_range, len(nn)))
    Gradgrad_dz3_int  = np.reshape(Gradgrad_dz3_internal, (N_internal,nt_range, len(nn)))
        
    ## Normal Solution from vertical particle velocity - 3D case
    body_est_TEST = np.zeros((N_internal,15,len(nn)))
    for fi in range(len(nn)):
       for i in range(N_internal):
          
                       Xl = []
                       Xl = np.append(gradDIV1_int[i,:,fi], gradDIV2_int[i,:,fi])
                       Xl = np.append(Xl, gradDIV3_int[i,:,fi])

                       Xr = []
                       Xr = np.append(-rotROT1_int[i,:,fi]-(2*Gradgrad_dx3_int[i,:,fi]*(dx**3)), -rotROT2_int[i,:,fi]-(2*Gradgrad_dy3_int[i,:,fi]*(dy**3)))
                       Xr = np.append(Xr, -rotROT3_int[i,:,fi]-(2*Gradgrad_dz3_int[i,:,fi]*(dz**3)))

                       ADD_vs_x_up = []
                       ADD_vs_x_up = np.append(-divU_FD_dx_int[i,:,fi]+Gradgrad_dx3_int[i,:,fi],np.zeros((nt_range))+10e-20)
                       ADD_vs_x_up = np.append(ADD_vs_x_up, np.zeros((nt_range))+10e-20)
                                      
                       ADD_vs_y_up = []
                       ADD_vs_y_up = np.append(np.zeros((nt_range))+10e-20,-divU_FD_dy_int[i,:,fi]+Gradgrad_dy3_int[i,:,fi])
                       ADD_vs_y_up = np.append(ADD_vs_y_up, np.zeros((nt_range))+10e-20)
                                      
                       ADD_vs_z_up = []
                       ADD_vs_z_up = np.append(np.zeros((nt_range))+10e-20,np.zeros((nt_range))+10e-20)
                       ADD_vs_z_up = np.append(ADD_vs_z_up,-divU_FD_dz_int[i,:,fi]+Gradgrad_dz3_int[i,:,fi]) 

                       ADD_vs_x_down = []
                       ADD_vs_x_down = np.append(divU_FD_dx_int[i,:,fi]-Gradgrad_dx3_int[i,:,fi],np.zeros((nt_range))+10e-20)
                       ADD_vs_x_down = np.append(ADD_vs_x_down, np.zeros((nt_range))+10e-20)
                                      
                       ADD_vs_y_down = []
                       ADD_vs_y_down = np.append(np.zeros((nt_range))+10e-20,divU_FD_dy_int[i,:,fi]-Gradgrad_dy3_int[i,:,fi])
                       ADD_vs_y_down = np.append(ADD_vs_y_down, np.zeros((nt_range))+10e-20)
                                      
                       ADD_vs_z_down = []
                       ADD_vs_z_down = np.append(np.zeros((nt_range))+10e-20,np.zeros((nt_range))+10e-20)
                       ADD_vs_z_down = np.append(ADD_vs_z_down, divU_FD_dz_int[i,:,fi]-Gradgrad_dz3_int[i,:,fi]) 
                                      
                       ADD_vp_x_up = []
                       ADD_vp_x_up = np.append(0.5*divU_FD_dx_int[i,:,fi],np.zeros((nt_range))+10e-20)
                       ADD_vp_x_up = np.append(ADD_vp_x_up, np.zeros((nt_range))+10e-20)
                                      
                       ADD_vp_y_up = []
                       ADD_vp_y_up = np.append(np.zeros((nt_range))+10e-20,0.5*divU_FD_dy_int[i,:,fi])
                       ADD_vp_y_up = np.append(ADD_vp_y_up, np.zeros((nt_range))+10e-20)
                                      
                       ADD_vp_z_up = []
                       ADD_vp_z_up = np.append(np.zeros((nt_range))+10e-20,np.zeros((nt_range))+10e-20)
                       ADD_vp_z_up = np.append(ADD_vp_z_up,0.5*divU_FD_dz_int[i,:,fi] )

                       ADD_vp_x_down = []
                       ADD_vp_x_down = np.append(-0.5*divU_FD_dx_int[i,:,fi],np.zeros((nt_range))+10e-20)
                       ADD_vp_x_down = np.append(ADD_vp_x_down, np.zeros((nt_range))+10e-20)
                                      
                       ADD_vp_y_down = []
                       ADD_vp_y_down = np.append(np.zeros((nt_range))+10e-20,-0.5*divU_FD_dy_int[i,:,fi])
                       ADD_vp_y_down = np.append(ADD_vp_y_down, np.zeros((nt_range))+10e-20)
                                      
                       ADD_vp_z_down = []
                       ADD_vp_z_down = np.append(np.zeros((nt_range))+10e-20,np.zeros((nt_range))+10e-20)
                       ADD_vp_z_down = np.append(ADD_vp_z_down, -0.5*divU_FD_dz_int[i,:,fi] )

                       X = np.array([np.ones((3*nt_range)),ADD_vs_x_up,ADD_vs_y_up,ADD_vs_z_up, ADD_vs_x_down, ADD_vs_y_down, ADD_vs_z_down, ADD_vp_x_up,ADD_vp_y_up,ADD_vp_z_up,\
                                     ADD_vp_x_down, ADD_vp_y_down, ADD_vp_z_down, Xl, Xr]).T
                       y_rh = []
                       y_rh=np.append(dttv1_int[i,:,fi],dttv2_int[i,:,fi])
                       y_rhs=np.append(y_rh,dttv3_int[i,:,fi])


                       inv = np.linalg.inv(np.dot(X.T,X))
                       XX = np.dot(inv, X.T)
                       SOL = np.dot(XX, y_rhs)
                       body_est_TEST[i,:,fi] = np.sqrt(abs(SOL))
                    

    vs_est_3eq_INHOM = np.reshape(body_est_TEST[:,13], (nrx-2,nry-2,nrz-2, len(nn)))
    vp_est_3eq_INHOM = np.reshape(body_est_TEST[:,14], (nrx-2,nry-2,nrz-2, len(nn)))
                                                 
    vs_x_up_INHOM = np.reshape(body_est_TEST[:,1], (nrx-2,nry-2,nrz-2, len(nn)))
    vs_y_up_INHOM = np.reshape(body_est_TEST[:,2], (nrx-2,nry-2,nrz-2, len(nn)))
    vs_z_up_INHOM = np.reshape(body_est_TEST[:,3], (nrx-2,nry-2,nrz-2, len(nn)))
    vs_x_down_INHOM = np.reshape(body_est_TEST[:,4], (nrx-2,nry-2,nrz-2, len(nn)))
    vs_y_down_INHOM = np.reshape(body_est_TEST[:,5], (nrx-2,nry-2,nrz-2, len(nn)))
    vs_z_down_INHOM = np.reshape(body_est_TEST[:,6], (nrx-2,nry-2,nrz-2, len(nn)))       
    vp_x_up_INHOM = np.reshape(body_est_TEST[:,7], (nrx-2,nry-2,nrz-2, len(nn)))
    vp_y_up_INHOM = np.reshape(body_est_TEST[:,8], (nrx-2,nry-2,nrz-2, len(nn)))
    vp_z_up_INHOM = np.reshape(body_est_TEST[:,9], (nrx-2,nry-2,nrz-2, len(nn)))
    vp_x_down_INHOM = np.reshape(body_est_TEST[:,10], (nrx-2,nry-2,nrz-2, len(nn)))
    vp_y_down_INHOM = np.reshape(body_est_TEST[:,11], (nrx-2,nry-2,nrz-2, len(nn)))
    vp_z_down_INHOM = np.reshape(body_est_TEST[:,12], (nrx-2,nry-2,nrz-2, len(nn)))                                                    
    sc_term_INHOM = np.reshape(body_est_TEST[:,0], (nrx-2,nry-2,nrz-2, len(nn)))
    
    
    
    return vp_est_3eq_INHOM, vs_est_3eq_INHOM, vs_x_up_INHOM, vs_y_up_INHOM, vs_z_up_INHOM, vs_x_down_INHOM, vs_y_down_INHOM, vs_z_down_INHOM,\
           vp_x_up_INHOM, vp_y_up_INHOM, vp_z_up_INHOM, vp_x_down_INHOM, vp_y_down_INHOM, vp_z_down_INHOM, sc_term_INHOM

# +
# def NORM_SOL_SYSTEM_INHOM_FORWARD_FD_3D_ELASTIC(dttv1_internal, dttv2_internal, dttv3_internal, rotROT1_internal, rotROT2_internal, rotROT3_internal,\
#                          gradDIV1_internal, gradDIV2_internal, gradDIV3_internal, nt_sub, nrx, nry, nrz, comp,order,\
#                                add1x_int, add1y_int,add1z_int,add2x_int,add2y_int,add2z_int,ADD2x_int,ADD2y_int,ADD2y_int):
        
#     add1x_int = np.reshape(add1x_internal, (N_internal,nt_range, len(nn)))
#     add1y_int = np.reshape(add1y_internal, (N_internal,nt_range, len(nn)))
#     add1z_int = np.reshape(add1z_internal, (N_internal,nt_range, len(nn)))
#     add2x_int = np.reshape(add2x_internal, (N_internal,nt_range, len(nn)))
#     add2y_int  = np.reshape(add2y_internal, (N_internal,nt_range, len(nn)))
#     add2z_int  = np.reshape(add2z_internal, (N_internal,nt_range, len(nn)))


#     ADD2x_int = np.reshape(ADD2x_internal, (N_internal,nt_range, len(nn)))
#     ADD2y_int  = np.reshape(ADD2y_internal, (N_internal,nt_range, len(nn)))
#     ADD2z_int  = np.reshape(ADD2z_internal, (N_internal,nt_range, len(nn)))

#     ## Normal Solution from vertical particle velocity - 3D case
#     body_est_TEST = np.zeros((N_internal,9,len(nn)))
#     for fi in range(len(nn)):
#        for i in range(N_internal):
#            #for ttt in range(1000,nt_sub):
#                        #X = np.array([[gradDIV1_int[i,:],-rotROT1_int[i,:]],[gradDIV2_int[i,:],-rotROT2_int[i,:]],\
#                        #             [gradDIV3_int[i,:],-rotROT3_int[i,:]]])
#                        Xl = []
#                        Xl = np.append(gradDIV1_int[i,:,fi]+add1x_int[i,:,fi], gradDIV2_int[i,:,fi]+add1y_int[i,:,fi])
#                        Xl = np.append(Xl, gradDIV3_int[i,:,fi]+add1z_int[i,:,fi])

#                        Xr = []
#                        Xr = np.append(-rotROT1_int[i,:,fi]-add2x_int[i,:,fi], -rotROT2_int[i,:,fi]-add2y_int[i,:,fi])
#                        Xr = np.append(Xr, -rotROT3_int[i,:,fi]-add2z_int[i,:,fi])

#     #                    ADD1 = []
#     #                    ADD1 = np.append(-add1x_int[i,:,fi],-add1y_int[i,:,fi])
#     #                    ADD1 = np.append(ADD1, -add1z_int[i,:,fi])
#                        ADD1 = []
#                        ADD1 = np.append(ADD2x_int[i,:,fi]+2*add1x_int[i,:,fi],ADD2x_int[i,:,fi])
#                        ADD1 = np.append(ADD1, ADD2x_int[i,:,fi])

#                        ADD2 = []
#                        ADD2 = np.append(ADD2y_int[i,:,fi],ADD2y_int[i,:,fi]+2*add1y_int[i,:,fi])
#                        ADD2 = np.append(ADD2, ADD2y_int[i,:,fi])

#     #                    ADD2 = []
#     #                    ADD2 = np.append(-add2x_int[i,:,fi]+ADD2x_int[i,:,fi],-add2y_int[i,:,fi]+ADD2y_int[i,:,fi])
#     #                    ADD2 = np.append(ADD2, -add2z_int[i,:,fi]+ADD2z_int[i,:,fi])
#                        ADD3 = []
#                        ADD3 = np.append(ADD2z_int[i,:,fi],ADD2z_int[i,:,fi])
#                        ADD3 = np.append(ADD3, ADD2z_int[i,:,fi]+2*add1z_int[i,:,fi])

#                        ADD4 = []
#                        ADD4 = np.append(-add1x_int[i,:,fi],np.zeros((nt_range)))
#                        ADD4 = np.append(ADD4, np.zeros((nt_range)))

#                        ADD5 = []
#                        ADD5 = np.append(np.zeros((nt_range)),-add1y_int[i,:,fi])
#                        ADD5 = np.append(ADD5, np.zeros((nt_range)))

#                        ADD6 = []
#                        ADD6 = np.append(np.zeros((nt_range)),np.zeros((nt_range)))
#                        ADD6 = np.append(ADD6, -add1z_int[i,:,fi])



#     #                    ADD3 = []
#     #                    ADD3 = np.append(-add2x_int[i,:,fi]+ADD2x_int[i,:,fi],-add2y_int[i,:,fi]+ADD2y_int[i,:,fi])
#     #                    ADD3 = np.append(ADD2, -add2z_int[i,:,fi]+ADD2z_int[i,:,fi])


#                        X = np.array([np.ones((3*nt_range)),ADD1,ADD2,ADD3,ADD4,ADD5,ADD6,Xl, Xr]).T
#                        #X= X.flatten()
#                        #X = np.reshape(X, (3,nt_range,2))
#                        #X = np.reshape(X,(3*nt_range,2)) #+ 1e-30
#                        #y_rhs = np.array([dttv1_int[i,:],dttv2_int[i,:],dttv3_int[i,:]])
#                        #y_rhs = y_rhs.flatten()
#                        #y_rhs = np.reshape(y_rhs,(3*nt_range))
#                        y_rh = []
#                        y_rh=np.append(dttv1_int[i,:,fi],dttv2_int[i,:,fi])
#                        y_rhs=np.append(y_rh,dttv3_int[i,:,fi])


#                        inv = np.linalg.inv(np.dot(X.T,X))
#                        XX = np.dot(inv, X.T)
#                        SOL = np.dot(XX, y_rhs)
#                        body_est_TEST[i,:,fi] = np.sqrt(abs(SOL))

#     vs_est_3eq_TEST = np.reshape(body_est_TEST[:,8], (nrx-2,nry-2,nrz-2, len(nn)))
#     vp_est_3eq_TEST = np.reshape(body_est_TEST[:,7], (nrx-2,nry-2,nrz-2, len(nn)))
#     est_3eq_TEST1 = np.reshape(body_est_TEST[:,1], (nrx-2,nry-2,nrz-2, len(nn)))
#     est_3eq_TEST2 = np.reshape(body_est_TEST[:,2], (nrx-2,nry-2,nrz-2, len(nn)))
#     est_3eq_TEST3 = np.reshape(body_est_TEST[:,3], (nrx-2,nry-2,nrz-2, len(nn)))
#     sc_term_TEST = np.reshape(body_est_TEST[:,0], (nrx-2,nry-2,nrz-2, len(nn)))
    
#     return vp_est_3eq_TEST,vs_est_3eq_TEST, est_3eq_TEST1, est_3eq_TEST2, est_3eq_TEST3, sc_term_TEST
# -

def NORM_SOL_3D_ELASTIC(dttv1_internal, dttv2_internal, dttv3_internal, rotROT1_internal, rotROT2_internal, rotROT3_internal,\
                        gradDIV1_internal, gradDIV2_internal, gradDIV3_internal, nt_sub, nrx, nry, nrz, comp,order):
    
    N_internal        = dttv3_internal[:,:,:,0].flatten().shape[0]
    dttv3_internal    = np.reshape(dttv3_internal, (N_internal,nt_sub))
    dttv2_internal    = np.reshape(dttv2_internal, (N_internal,nt_sub))
    dttv1_internal    = np.reshape(dttv1_internal, (N_internal,nt_sub))
    gradDIV3_internal = np.reshape(gradDIV3_internal, (N_internal,nt_sub))
    rotROT3_internal  = np.reshape(rotROT3_internal, (N_internal,nt_sub))
    gradDIV2_internal = np.reshape(gradDIV2_internal, (N_internal,nt_sub))
    rotROT2_internal  = np.reshape(rotROT2_internal, (N_internal,nt_sub))
    gradDIV1_internal = np.reshape(gradDIV1_internal, (N_internal,nt_sub))
    rotROT1_internal  = np.reshape(rotROT1_internal, (N_internal,nt_sub))
    
    if comp == 'X':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)), gradDIV1_internal[i,:],-rotROT1_internal[i,:]]).T
            y_rhs = dttv1_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-order,nry-order,nrz-order))
        vp_est = np.reshape(body_est[:,1], (nrx-order,nry-order,nrz-order))
        
    elif comp == 'Y':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)),gradDIV2_internal[i,:],-rotROT2_internal[i,:]]).T
            y_rhs = dttv2_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-order,nry-order,nrz-order))
        vp_est = np.reshape(body_est[:,1], (nrx-order,nry-order,nrz-order))    
    
    elif comp == 'Z':
        # Normal Solution from vertical particle velocity - 3D case
        body_est = np.zeros((N_internal,3))
        for i in range(N_internal):
            X = np.array([np.ones((nt_sub)),gradDIV3_internal[i,:],-rotROT3_internal[i,:]]).T
            y_rhs = dttv3_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est[i,:] = np.sqrt(abs(SOL))

        vs_est = np.reshape(body_est[:,2], (nrx-order,nry-order,nrz-order))
        vp_est = np.reshape(body_est[:,1], (nrx-order,nry-order,nrz-order))    
    
    return vp_est, vs_est


def NORM_SOL_2D_ELASTIC_OPTiii_no_dz(dttv1_internal, dttv2_internal, dttv3_internal, rotROT1_2D_internal, rotROT2_2D_internal,\
                                     rotROT3_2D_internal, gradDIV1_2D_internal, gradDIV2_2D_internal, gradDIV3_2D_internal,\
                                     nt_sub, nrx, nry, nrz, comp,order):
    
    N_internal        = dttv3_internal[:,:,:,0].flatten().shape[0]
    dttv3_internal    = np.reshape(dttv3_internal, (N_internal,nt_sub))
    dttv2_internal    = np.reshape(dttv2_internal, (N_internal,nt_sub))
    dttv1_internal    = np.reshape(dttv1_internal, (N_internal,nt_sub))

    gradDIV2_2D_internal = np.reshape(gradDIV2_2D_internal, (N_internal,nt_sub))
    rotROT2_2D_internal = np.reshape(rotROT2_2D_internal, (N_internal,nt_sub))
    gradDIV1_2D_internal = np.reshape(gradDIV1_2D_internal, (N_internal,nt_sub))
    rotROT1_2D_internal = np.reshape(rotROT1_2D_internal, (N_internal,nt_sub))
    gradDIV3_2D_internal = np.reshape(gradDIV3_2D_internal, (N_internal,nt_sub))
    rotROT3_2D_internal = np.reshape(rotROT3_2D_internal, (N_internal,nt_sub))

    # Normal Solution from vertical particle velocity - 2D case
    if comp=='X':
        body_est_2D = np.zeros((N_internal,2))
        for i in range(N_internal):
            X = np.array([gradDIV1_2D_internal[i,:],-rotROT1_2D_internal[i,:]]).T #+1e-30
            y_rhs = dttv1_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est_2D[i,:] = np.sqrt(abs(SOL))
            #print('Estimated vp velocity:',body_est_2D[i,0])
            #print('Estimated vs velocity:',body_est_2D[i,1])

        vs_est_2D = np.reshape(body_est_2D[:,1], (nrx-order,nry-order,nrz-order))
        vp_est_2D = np.reshape(body_est_2D[:,0], (nrx-order,nry-order,nrz-order))
        
    elif comp=='Y':
        body_est_2D = np.zeros((N_internal,2))
        for i in range(N_internal):
            X = np.array([gradDIV2_2D_internal[i,:],-rotROT2_2D_internal[i,:]]).T #+1e-30
            y_rhs = dttv2_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est_2D[i,:] = np.sqrt(abs(SOL))
            #print('Estimated vp velocity:',body_est_2D[i,0])
            #print('Estimated vs velocity:',body_est_2D[i,1])

        vs_est_2D = np.reshape(body_est_2D[:,1], (nrx-order,nry-order,nrz-order))
        vp_est_2D = np.reshape(body_est_2D[:,0], (nrx-order,nry-order,nrz-order))
        
    elif comp=='Z':
        body_est_2D = np.zeros((N_internal,2))
        for i in range(N_internal):
            X = np.array([gradDIV3_2D_internal[i,:],-rotROT3_2D_internal[i,:]]).T +1e-30
            y_rhs = dttv3_internal[i,:]#rhs_eq3#90*dttv3[a+4:b+4]#rhs3[a:b]

            inv = np.linalg.inv(np.dot(X.T,X))
            XX = np.dot(inv, X.T)
            SOL = np.dot(XX, y_rhs)
            body_est_2D[i,:] = np.sqrt(abs(SOL))
            #print('Estimated vp velocity:',body_est_2D[i,0])
            #print('Estimated vs velocity:',body_est_2D[i,1])

        vs_est_2D = np.reshape(body_est_2D[:,1], (nrx-order,nry-order,nrz-order))
        vp_est_2D = np.reshape(body_est_2D[:,0], (nrx-order,nry-order,nrz-order))

    return vp_est_2D, vs_est_2D
