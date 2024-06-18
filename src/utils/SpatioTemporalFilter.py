# import numpy as np
# import matplotlib.pyplot as plt
# import pickle, csv, pandas
# import math
# import cv2
# from StimtoArray import get_RDK
# from numba import jit
# from scipy import ndimage
#
# def f1(x,y,sigma_c = .35, sigma_g = .05):
#     alpha = np.arctan(x/sigma_c)
#     return (np.cos(alpha)**4)*(np.cos(4*alpha))*np.exp((-y**2)/(2*sigma_g**2))
#
# def f2(x,y,sigma_c = .35, sigma_g = .05):
#     alpha = np.arctan(x/sigma_c)
#     return (np.cos(alpha)**4)*(np.sin(4*alpha))*np.exp((-y**2)/(2*sigma_g**2))
#
# def g1(t):
#     return ((60 * t) ** 3) * (np.exp(-60 * t)) * ((1 / np.math.factorial(3)) - (((60 * t) ** 2) / np.math.factorial(3 + 2)))
#
# def g2(t):
#     return ((60 * t) ** 5) * (np.exp(-60 * t)) * ((1 / np.math.factorial(5)) - (((60 * t) ** 2) / np.math.factorial(5 + 2)))
#
# def create_filters(savename = 'spatiotemporal_filters',nFrames=360):
#     x_space = np.arange(-.7, .7, .05)
#     y_space = np.arange(-.7, .7, .05)
#     t_space = np.arange(0, .2, .01)
#     # t_space = np.linspace(0,1000,nFrames)/1000
#
#     # Linear Spatiotemporal Filter (XT) at specific Y value
#     Filter_XT_left1 = np.empty((len(t_space),len(x_space),len(y_space)))
#     Filter_XT_left2 = np.empty((len(t_space),len(x_space),len(x_space)))
#     Filter_XT_right1 = np.empty((len(t_space),len(x_space),len(x_space)))
#     Filter_XT_right2 = np.empty((len(t_space),len(x_space),len(x_space)))
#     for i,t in enumerate(t_space):
#         for j,x in enumerate(x_space):
#             Filter_XT_right1[i,j,:] = f1(x,y_space)*g1(t) + f2(x,y_space)*g2(t)
#             Filter_XT_right2[i,j,:] = f2(x,y_space)*g1(t) - f1(x,y_space)*g2(t)
#             Filter_XT_left1[i,j,:] = f1(x,y_space)*g1(t) - f2(x,y_space)*g2(t)
#             Filter_XT_left2[i,j,:] = f2(x,y_space)*g1(t) + f1(x,y_space)*g2(t)
#
#     # # Spatial even and off functions
#     # plt.plot(x_space,f1(x_space,y_space))
#     # plt.plot(x_space,f2(x_space,y_space))
#     # plt.show()
#
#     # # Temporal functions
#     # plt.plot(t_space,g1(t_space))
#     # plt.plot(t_space,g2(t_space))
#     # plt.show()
#
#     # fig2, ax = plt.subplots(2,2)
#     # ax[0][0].imshow(Filter_XT_right1,'gray')
#     # ax[1][0].imshow(Filter_XT_right2,'gray')
#     # ax[0][1].imshow(Filter_XT_left1,'gray')
#     # ax[1][1].imshow(Filter_XT_left2,'gray')
#     # fig2.show()
#
#     # Saving the Filters:
#     with open(savename+'.pkl', 'wb') as f:
#         pickle.dump([Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2], f)
#
#     return Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2
#
# def create_stimulus(Coh,filename='RDK_position_matrix_processed'):
#     RDK_processed = np.zeros((len(Coh), frames, 192, 108))
#     for i_coh, coh in enumerate(Coh):
#         RDK = get_RDK(coh=coh,seed=1,nFrames=frames)
#
#         RDK /= np.max(RDK)  # Converting and normalizing matrix for grayscale image
#         for t in range(frames):
#             RDK_processed[i_coh, t,:,:] = cv2.resize(RDK[t,:,:],(108,192))
#
#     np.savez(filename+'.npz',RDK_processed=RDK_processed,Coh=Coh)
#     return  RDK_processed, Coh
#
#
# # @jit(nopython=True,parallel=True)
# def Calculate_ME(RDK_processed,Coh):
#
#     for i, coh in enumerate(Coh):
#         # RDK = get_RDK(coh=coh,seed=1,nFrames=frames)
#         #
#         # RDK /= np.max(RDK)  # Converting and normalizing matrix for grayscale image
#         # RDK_processed = np.zeros((frames,192,108))
#         # for i in range(frames):
#         #     RDK_processed[i,:,:] = cv2.resize(RDK[i,:,:],(108,192))
#
#         ## Convolution
#         ME_right1 = ndimage.convolve(RDK_processed[i],Filter_XT_right1, mode='constant', cval=0.0)
#         ME_right2 = ndimage.convolve(RDK_processed[i],Filter_XT_right2, mode='constant', cval=0.0)
#         ME_left1 = ndimage.convolve(RDK_processed[i],Filter_XT_left1, mode='constant', cval=0.0)
#         ME_left2 = ndimage.convolve(RDK_processed[i],Filter_XT_left2, mode='constant', cval=0.0)
#
#         ME_RIGHT = (ME_right1**2) + (ME_right2**2)
#         ME_LEFT = (ME_left1**2) + (ME_left2**2)
#         ME = ME_RIGHT - ME_LEFT
#
#         ME_XT = np.mean(ME,axis=2)
#         ME_T = np.mean(ME_XT,axis=1)
#
#         ME_T_ms[i,:] = np.repeat(ME_T,16)
#
#
# import multiprocessing
# if __name__ == '__main__':
#
#     # ## Create or import filters
#     # response = input("Create Filters?")
#     # yes = ['Y', 'y', 'YES', 'Yes', 'yes']
#     # # If new filters need to be created
#     # if response in yes:
#     #     filename = input("Save Filters as: ")
#     #     if not filename:
#     #         filename = 'spatiotemporal_filters'
#     #     frames = input("number of Frames: ")
#     #     if not frames:
#     #         frames = 360
#     #     Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2 = create_filters(filename, frames)
#     #
#     # else:
#     #     filename = input("Import Filters file. Filename: ")
#     #     if not filename:
#     #         with open('spatiotemporal_filters.pkl', 'rb') as f:
#     #             Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2 = pickle.load(f)
#     #     else:
#     #         with open(filename + '.pkl', 'rb') as f:
#     #             Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2 = pickle.load(f)
#     #
#     #     frames = input("number of Frames: ")
#     #     if not frames:
#     #         frames = 360
#     #
#     #
#     # ## Create/Import Stimulus Array
#     # response = input("Create Stimulus?")
#     # # If new filters need to be created
#     # if response in yes:
#     #     filename = input("Save Positions as: ")
#     #     if not filename:
#     #         filename = 'RDK_position_matrix_processed'
#     #
#     #     Coh = input("Insert Coherence list separated by ',': ")
#     #     if not Coh:
#     #         Coh = [-100, -70, -36, -18, -9, 0, 9, 18, 36, 70, 100]
#     #     else:
#     #         Coh = list(map(int,Coh.split(',')))
#     #     RDK_processed, Coh = create_stimulus(Coh=Coh, filename=filename)
#     #
#     # else:
#     #     filename = input("Import RDK position. Filename: ")
#     #     if not filename:
#     #         filename = 'RDK_position_matrix_processed'
#     #
#     #     temp = np.load(filename+'.npz')
#     #     RDK_processed, Coh = temp['RDK_processed'], temp['Coh']
#
#     with open('spatiotemporal_filters.pkl', 'rb') as f:
#         Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2 = pickle.load(f)
#     temp = np.load('RDK_position_matrix_processed' + '.npz')
#     RDK_processed, Coh = temp['RDK_processed'], temp['Coh']
#     frames = 360
#
#     global ME_T_ms;
#     ME_T_ms = np.zeros((len(Coh), frames * 16))
#
#
#     Calculate_ME(RDK_processed,Coh)
#     # pool = multiprocessing.Pool(processes=11)
#     # pool.map(Calculate_ME, range(11))
#     # pool.close()
#     # pool.join()
#     # print('done')
#
#
#     import matplotlib.pyplot as plt
#
#     for i, coh in enumerate(Coh):
#         plt.plot(ME_T_ms[i,:],label=str(coh))
#     plt.legend(loc='best')
#     plt.show()
#
#     np.savez('Motion_Energy_all_normal.npz', Motion_Energy = ME_T_ms, RDK_processed=RDK_processed, Coh=Coh)
#
#
#     pass
#
#
#
#






## RDK with pulse
import numpy as np
import matplotlib.pyplot as plt
import pickle, csv, pandas
import math
import cv2
from StimtoArray import get_RDK
from numba import jit
from scipy import ndimage

def f1(x,y,sigma_c = .35, sigma_g = .05):
    alpha = np.arctan(x/sigma_c)
    return (np.cos(alpha)**4)*(np.cos(4*alpha))*np.exp((-y**2)/(2*sigma_g**2))

def f2(x,y,sigma_c = .35, sigma_g = .05):
    alpha = np.arctan(x/sigma_c)
    return (np.cos(alpha)**4)*(np.sin(4*alpha))*np.exp((-y**2)/(2*sigma_g**2))

def g1(t):
    return ((60 * t) ** 3) * (np.exp(-60 * t)) * ((1 / np.math.factorial(3)) - (((60 * t) ** 2) / np.math.factorial(3 + 2)))

def g2(t):
    return ((60 * t) ** 5) * (np.exp(-60 * t)) * ((1 / np.math.factorial(5)) - (((60 * t) ** 2) / np.math.factorial(5 + 2)))

def create_filters(savename = 'spatiotemporal_filters',nFrames=360):
    x_space = np.arange(-.7, .7, .05)
    y_space = np.arange(-.7, .7, .05)
    t_space = np.arange(0, .2, .01)
    # t_space = np.linspace(0,1000,nFrames)/1000

    # Linear Spatiotemporal Filter (XT) at specific Y value
    Filter_XT_left1 = np.empty((len(t_space),len(x_space),len(y_space)))
    Filter_XT_left2 = np.empty((len(t_space),len(x_space),len(x_space)))
    Filter_XT_right1 = np.empty((len(t_space),len(x_space),len(x_space)))
    Filter_XT_right2 = np.empty((len(t_space),len(x_space),len(x_space)))
    for i,t in enumerate(t_space):
        for j,x in enumerate(x_space):
            Filter_XT_right1[i,j,:] = f1(x,y_space)*g1(t) + f2(x,y_space)*g2(t)
            Filter_XT_right2[i,j,:] = f2(x,y_space)*g1(t) - f1(x,y_space)*g2(t)
            Filter_XT_left1[i,j,:] = f1(x,y_space)*g1(t) - f2(x,y_space)*g2(t)
            Filter_XT_left2[i,j,:] = f2(x,y_space)*g1(t) + f1(x,y_space)*g2(t)

    # # Spatial even and off functions
    # plt.plot(x_space,f1(x_space,y_space))
    # plt.plot(x_space,f2(x_space,y_space))
    # plt.show()

    # # Temporal functions
    # plt.plot(t_space,g1(t_space))
    # plt.plot(t_space,g2(t_space))
    # plt.show()

    # fig2, ax = plt.subplots(2,2)
    # ax[0][0].imshow(Filter_XT_right1,'gray')
    # ax[1][0].imshow(Filter_XT_right2,'gray')
    # ax[0][1].imshow(Filter_XT_left1,'gray')
    # ax[1][1].imshow(Filter_XT_left2,'gray')
    # fig2.show()

    # Saving the Filters:
    with open(savename+'.pkl', 'wb') as f:
        pickle.dump([Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2], f)

    return Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2

def create_stimulus(Coh,pulse_coh,pulse_dur,pulse_t,filename='RDKPulse_position_matrix_processed',):
    RDK_processed = np.zeros((len(Coh), frames, 192, 108))
    for i_coh, coh in enumerate(Coh):

        RDK = get_RDK(coh=coh,seed=1,nFrames=frames,pulse_coh=pulse_coh[i_coh], pulse_t=pulse_t[i_coh], pulse_dur=pulse_dur[i_coh])

        RDK /= np.max(RDK)  # Converting and normalizing matrix for grayscale image
        for t in range(frames):
            RDK_processed[i_coh, t,:,:] = cv2.resize(RDK[t,:,:],(108,192))

    np.savez(filename+'.npz',RDK_processed=RDK_processed,Coh=Coh)
    return  RDK_processed, Coh


# @jit(nopython=True,parallel=True)
def Calculate_ME(RDK_processed,Coh):

    for i, coh in enumerate(Coh):
        # RDK = get_RDK(coh=coh,seed=1,nFrames=frames)
        #
        # RDK /= np.max(RDK)  # Converting and normalizing matrix for grayscale image
        # RDK_processed = np.zeros((frames,192,108))
        # for i in range(frames):
        #     RDK_processed[i,:,:] = cv2.resize(RDK[i,:,:],(108,192))

        ## Convolution
        ME_right1 = ndimage.convolve(RDK_processed[i],Filter_XT_right1, mode='constant', cval=0.0)
        ME_right2 = ndimage.convolve(RDK_processed[i],Filter_XT_right2, mode='constant', cval=0.0)
        ME_left1 = ndimage.convolve(RDK_processed[i],Filter_XT_left1, mode='constant', cval=0.0)
        ME_left2 = ndimage.convolve(RDK_processed[i],Filter_XT_left2, mode='constant', cval=0.0)

        ME_RIGHT = (ME_right1**2) + (ME_right2**2)
        ME_LEFT = (ME_left1**2) + (ME_left2**2)
        ME = ME_RIGHT - ME_LEFT

        ME_XT = np.mean(ME,axis=2)
        ME_T = np.mean(ME_XT,axis=1)

        ME_T_ms[i,:] = np.repeat(ME_T,16)

    return ME_T_ms, ME_T


import multiprocessing
if __name__ == '__main__':
    frames = 360
    # ## Create or import filters
    # response = input("Create Filters?")
    yes = ['Y', 'y', 'YES', 'Yes', 'yes']
    # # If new filters need to be created
    # if response in yes:
    #     filename = input("Save Filters as: ")
    #     if not filename:
    #         filename = 'spatiotemporal_filters'
    #     frames = input("number of Frames: ")
    #     if not frames:
    #         frames = 360
    #     Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2 = create_filters(filename, frames)
    #
    # else:
    #     filename = input("Import Filters file. Filename: ")
    #     if not filename:
    #         with open('spatiotemporal_filters.pkl', 'rb') as f:
    #             Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2 = pickle.load(f)
    #     else:
    #         with open(filename + '.pkl', 'rb') as f:
    #             Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2 = pickle.load(f)
    #
    #     frames = input("number of Frames: ")
    #     if not frames:
    #         frames = 360
    #
    #
    ## Create/Import Stimulus Array
    response = input("Create Stimulus?")
    # If new filters need to be created
    if response in yes:
        filename = input("Save Positions as: ")
        if not filename:
            filename = 'RDKPulseFixed_position_matrix_processed'

        Coh = input("Insert Coherence list separated by ',': ")
        if not Coh:
            Coh = [-100, -70, -36, -18, -9, 0, 9, 18, 36, 70, 100]
        else:
            Coh = list(map(int,Coh.split(',')))

        pulse_coh = input("Insert Pulse Coherence list separated by ',': ")
        if not pulse_coh:
            pulse_coh = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
            # pulse_coh = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        else:
            pulse_coh = list(map(int,pulse_coh.split(',')))

        pulse_t = input("Insert Pulse time list separated by ',': ")
        if not pulse_t:
            pulse_t = np.ones(len(Coh))*150
            # pulse_t = np.random.randint(50,300,len(Coh))
        else:
            pulse_t = list(map(int,pulse_t.split(',')))

        pulse_dur = input("Insert Pulse Duration list separated by ',': ")
        if not pulse_dur:
            pulse_dur = np.ones(len(Coh))*30
        else:
            pulse_dur = list(map(int,pulse_dur.split(',')))

        RDK_processed, Coh = create_stimulus(Coh=Coh, filename=filename, pulse_coh=pulse_coh, pulse_dur=pulse_dur, pulse_t=pulse_t)

    else:
        filename = input("Import RDK position. Filename: ")
        if not filename:
            filename = 'RDKPulseFixed_position_matrix_processed'

        temp = np.load(filename+'.npz')
        RDK_processed, Coh = temp['RDK_processed'], temp['Coh']

    with open('spatiotemporal_filters.pkl', 'rb') as f:
        Filter_XT_right1, Filter_XT_right2, Filter_XT_left1, Filter_XT_left2 = pickle.load(f)
    # temp = np.load('RDK_position_matrix_processed' + '.npz')
    # RDK_processed, Coh = temp['RDK_processed'], temp['Coh']
    frames = 360

    global ME_T_ms;
    ME_T_ms = np.zeros((len(Coh), frames * 16))


    ME_T_ms, ME_T = Calculate_ME(RDK_processed,Coh)


    np.savez('Motion_Energy_all_pulse_fixed.npz', Motion_Energy = ME_T_ms, Motion_Energy_frames = ME_T, RDK_processed=RDK_processed, Coh=Coh, Pulse_Dur = pulse_dur, Pulse_time = pulse_t, Pulse_coh = pulse_coh)

    import matplotlib.pyplot as plt

    for i, coh in enumerate(Coh):
        plt.plot(ME_T_ms[i, :], label=str(coh))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig('Motion_Energy_all_pulse_fixed_500ms.png',dpi=600, bbox_inches='tight')
    # plt.show()

    pass





##
# import matplotlib.pyplot as plt
#
# for i in range(360):
#     plt.imshow(RDK_processed[5,i,:],'gray')
#     plt.pause()