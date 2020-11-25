import sys
import os
import numpy as np
import nibabel as nib
from nilearn.image import coord_transform,smooth_img,index_img,math_img
from nilearn.masking import apply_mask,unmask
from nilearn import plotting as niplt
from nilearn.datasets import load_mni152_template,load_mni152_brain_mask
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import inv
import argparse



def import_data(filename):
    """Given the path to the mat file imports position time and functional value"""
    
    
    datas = loadmat(filename)
    
    position   = datas['sdDIFF_FR_C'][0][0][2]
    time  = datas['sdDIFF_FR_C'][0][0][0]
    functional   = datas['sdDIFF_FR_C'][0][0][5][0][0][1]

    return position,time[0],functional


def plot_2D_crosscuts(bm_data,coord,fdir):
    
    # Get the locations of the mni template
    XX,YY,ZZ = np.where(bm_data > 0.1)
    # Transform a few of them in MNI space
    nskip = 29
    xmni,ymni,zmni = coord_transform(XX.ravel()[::nskip],
                           YY.ravel()[::nskip],
                           ZZ.ravel()[::nskip],bm.affine)
    
    # Plot the template and the fieldtrip locations together in a 2D crosscuts way
    fig,ax = plt.subplots(ncols = 3, figsize = (15,5))
    ax[0].scatter(xmni,ymni, s = 1)
    ax[1].scatter(xmni,zmni, s = 1)
    ax[2].scatter(ymni,zmni, s = 1)

    # ax[0].scatter(XX.ravel()[::nskip],YY.ravel()[::nskip], s = bm_data.ravel()[::nskip])
    # ax[1].scatter(XX.ravel()[::nskip],ZZ.ravel()[::nskip], s = bm_data.ravel()[::nskip])
    # ax[2].scatter(YY.ravel()[::nskip],ZZ.ravel()[::nskip], s = bm_data.ravel()[::nskip])

    crosscuts = [[0,1],[0,2],[1,2]]

    for i,cc in enumerate(crosscuts):

        ax[i].scatter(coord[:,cc[0]],coord[:,cc[1]],
                      c = 'C1', alpha = 0.5, s = 1)

        ax[i].set_title('{} crosscut'.format(cc))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
    plt.savefig(fdir + '2Dcrosscut.png',fmt = 'png', dpi = 300)
    plt.close('all')
    
    return

def plot_sources_on_brain(loc_volume,bm,bm_data):
    my_brain = np.zeros_like(bm_data)
    my_brain[loc_volume[:,0],loc_volume[:,1],loc_volume[:,2]] = 1
    img = nib.Nifti2Image(my_brain,affine = bm.affine)
    niplt.view_img(img, cmap='cyan_orange')
                          
    return

def generate_nifti(loc_volume,av_win,bm,smooth_f,fdir):

    fun_data = np.zeros((bm.shape[0],bm.shape[1],bm.shape[2],t_size))
    for iv,(xv,yv,zv) in enumerate(loc_volume):
        # Put the time average of the functional data over a time window av_win long
        fun_data[xv,yv,zv,:] = np.average(fun[iv,:].reshape((-1,av_win)), axis = 1)

    fun_img = nib.Nifti2Image(fun_data,affine = bm.affine)

    # Apply smoothing
    fun_img = smooth_img(fun_img,smooth_f)
    # Apply bm mask
    fun_img = unmask(apply_mask(fun_img,bm),bm)
    
    
    fout = fdir + 'T{:03d}_S{:02d}.nii.gz'.format(av_win,smooth_f)
    print('Nifti saved in {}'.format(fout))
    
    nib.save(fun_img, fout)

    return fun_img
    
    
    
    
if __name__ == '__main__':

    ### PARSING INPUT ARGUMENTS #######
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='filename',
                        help='The full path to the input .mat file to analyze')
    parser.add_argument('-T', action='store', dest='av_win', default = 256,
                        help='The size of the window for temporal averaging (in eeg timepoints, must be a power of 2)')
    parser.add_argument('-S', action='store', dest='smooth_f', default = 5,
                        help='The FWHM (in mm) for the spatial smoothing to be applied to the reconstructed signal')
    results = parser.parse_args()
    fname = results.filename
    smooth_f = results.smooth_f
    av_win = results.av_win
    if np.log2(av_win)%1!=0:
        print('You did not input a power of 2 for the temporal averaging window')
        av_win = int(2**np.round(np.log2(av_win)))
        print('Average window approximated to the nearest power of 2: av_win = {}'.format(av_win))
    ######################################
    fdir = os.path.split(fname)[0] + '/'

    # Import the eeg data
    pos,time,fun = import_data(fname)


    # Import the mni template
    bm = load_mni152_brain_mask()
    bm_data = bm.get_fdata()
    Lx,Ly,Lz = bm_data.shape
    
    
    # Apply shift and scaling to the data and reorder the axis in pos
    shift = np.array([0,40,40])
    scale = np.array([1,1,1])
    ix,iy,iz = [1,0,2]
    pos_ref = np.array([pos[:,ix],pos[:,iy],pos[:,iz]]).T
    transformed_pos = pos_ref * scale[None,:] - shift[None,:]
    
    # Plot a 2d crosscut overview to check that the scaling makes sense
    plot_2D_crosscuts(bm_data,transformed_pos,fdir)
    
    # Send the position to volume space
    pos_volume = coord_transform(transformed_pos[:,0],
                                 transformed_pos[:,1],
                                 transformed_pos[:,2],
                                 inv(bm.affine))
    loc_volume = np.round(pos_volume).astype(int).T
   
    # Plot the location of source detection on a brain template
#     plot_sources_on_brain(loc_volume,bm,bm_data)
    
    
    # Define the window for temporal averaging
    dt = (time[1] - time[0]) * av_win
    time_av = np.arange(time[0],time[-1],dt)
    t_size = len(time_av)
    print('The software is currently averaging the time traces over a time window of {} s'.format(dt))
    print('This will result in a nifti file with {} volumes.'.format(t_size))

    
    # Generate the nifti(s)
    fun_img = generate_nifti(loc_volume,av_win,bm,smooth_f,fdir)
        
