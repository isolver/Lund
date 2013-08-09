'''Simple program for finding a pupil and calibrating eye positions.

- Finds a pupil center
- indicates the center
- generates an outplot
- saves the data
- loads the data
- calibrates the eye position data

Author: Thomas Haslwanter and Erich Schneider
Date:   August 2013
Ver:    1.1

'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.ndimage.measurements as spm

from numpy import *
from scipy.stats import *
from scipy.io import loadmat, savemat
from scipy.cluster.vq import kmeans
from scipy import signal


def diff3(x):
    dxdt = zeros(x.shape)
    dxdt[1:-1, :] = 0.5*(x[2:, :] - x[:-2, :])
    dxdt[[0, -1], :] = dxdt[[1, -2], :]
    return dxdt


def adaptThreshold(img, thresh):
    ''' Adapt threshold '''
    # defaults for video1: h*w/50, h*w/4
    h,w = img.shape
    lower = h*w/4
    upper = h*w/3

    bw = img < thresh

    while np.sum(bw) < lower:
        thresh += 5
        bw  = img<thresh
    while np.sum(bw) > upper:
        thresh -= 5
        bw  = img<thresh

    return (bw, thresh)


def findCenter(frame, threshold):    
    '''Take a frame, and find the pupil center'''
    img = frame[...,0]
    h,w = img.shape
    
    # Threshold the image and adapt - if necessary - the threshold
    (bw, threshold) = adaptThreshold(img, threshold)
        
    # Flip b/w, and convert to uint8
    im_bw = np.array(~bw, dtype=np.uint8)*255
    
    algorithmNr = 0
    
    if algorithmNr == 0:
        labelled_array, num_features = spm.label(im_bw)
        sizes = spm.sum(bw, labelled_array, range(num_features))
        center = spm.center_of_mass(bw, labelled_array, np.argmax(sizes))    
        center = [center[1], center[0]]
    
    elif algorithmNr == 1:
        # Fill the corners
        # Note that the mask has to be 2 pixels larger than the image!
        mask = np.zeros( (h+2,w+2), dtype=np.uint8)
        cv2.floodFill(im_bw, mask, (1,h-1), 255)
        
        # After the floodfill operation, the "mask" indicates which areas have
        # been filled. It therefore has to be re-set for the next filling task.
        mask = np.zeros( (h+2,w+2), dtype=np.uint8)
        cv2.floodFill(im_bw, mask, (w-1,h-1), 255)
        
        # Fill the holes in the pupil
        wHoles = im_bw.copy()
        mask = np.zeros( (h+2,w+2), dtype=np.uint8)
        cv2.floodFill(wHoles, mask, (1,1), 0)
        im_bw[wHoles==255] = 0
        
        # Close the image, to remove the eye-brows    
        radius = 25
        strEl = np.zeros((2*radius+1, 2*radius+1), dtype=np.uint8)
        cv2.circle(strEl, (radius,radius), radius, 255, -1)
        
        closed = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, strEl)
        
        # find the edge
        edgeThresholds = (1000, 1000)
        edge = cv2.Canny(closed, edgeThresholds[0], edgeThresholds[1], apertureSize=5)
        
        # find the center of the edge
        edge_y, edge_x = np.where(edge==255)
        center = np.array([np.mean(edge_x), np.mean(edge_y)])
        
    if np.any(np.isnan(center)):
        center = np.zeros(2)
        
    return (center, threshold)
        

def heatMap (xy):
    # First calculate the 2D histogram
    hist2d, xedges, yedges = np.histogram2d(xy[:,1], xy[:,0], bins=(100, 100))
    # Then blur the histogram
    filtx = signal.gaussian(31, 31/6.0)
    filtx = filtx/sum(filtx)
    filty = filtx
    hist2d = signal.sepfir2d(hist2d, filtx, filty)

    # Calculate the range / extent    
    extent = array([amin(xy[:,0]), amax(xy[:,0]), amin(xy[:,1]), amax(xy[:,1])])
    return (hist2d, extent)


def plotAndCalibration(time, centerPupil):
    ''' Plot the data as pts, and then calibrate the data from a pre-recorded calibration-file'''
    
    # Subplot 1: hor/ver pupil position, in points -----------------------------------------------
    plt.cla()
    plt.subplot(2,2,1)
    plt.plot(time, centerPupil[:,0], label='horizontal')
    plt.hold(True)
    plt.plot(time, centerPupil[:,1], label='vertical')
    plt.legend()
    plt.ylabel('centerPupil [pts]')
    plt.hold(False)
    
    # ------------- The rest of this function plots pre-recorded data -------------------
    
    # Load the pupil positions from a Matlab calibration file
    datafile = 'p00006s001_calib.mat'    
    Data = loadmat(datafile, chars_as_strings=True)
    SamplingRate = 220.0     # The original sampling rate is 220 Hz
    
    # Extract the arrays that we need
    xy = Data['LeftPupilPx']
    calibmat = Data['LeftEyeCal']
    
    # Calculate the range / extent    
    t = np.arange(xy.shape[0])/SamplingRate
    
    # Plot Nr 3: x/y view of pre-recorded data in pixel, together with clustering information -------------
    
    # Heat map; This is just for fun
    hist2d, ex = heatMap (xy)
    
    # First plot the heat map ...
    plt.subplot(2,2,3)
    plt.cla()
    plt.hold(True)
    plt.imshow(hist2d, origin='lower', extent=ex, cmap='afmhot')
    plt.contour(hist2d, origin='lower', extent=ex)
    
    # ... and then the data and the clustering

    # Cluster analysis of the fixations
    # Cluster analysis of the fixations
    ntarget = 5;
    calibtarget = 8.5; # Laser dots are 8.5 deg apart
    clust, distortion = kmeans (xy, ntarget)
    
    plt.plot(xy[:,0], xy[:,1], '0.8', clust[:,0], clust[:,1], 'ro');
    plt.ylabel('Vertical [Pixel]')
    plt.xlabel('Horizontal [Pixel]')
    plt.hold(False)
    
    # Define fixations
    Cr0 = array(
            [[0, 0],
             [0,-1],
             [0, 1],
             [-1,0],
             [1, 0]])
       
    imax = np.nanargmax(clust, axis=0)
    imin = np.nanargmin(clust, axis=0)
    # Indices of secondary fixations
    iorder = np.array([imax[1], imin[1], imin[0], imax[0]])
    # Index of primary fixation
    imid = setdiff1d(arange(ntarget),iorder)
    ic0 = concatenate((imid,iorder)) 
    C = clust[ic0,:]
    
    Cr_deg = -Cr0*calibtarget;
    Cr = tan(-Cr_deg * pi/180)
    Ce = concatenate((C, ones((C.shape[0],1))), axis=1)
    
    # Linear fit; In Matlab it's Cr.T / Ce.T
    Ch = np.linalg.lstsq(Ce, Cr[:,0])[0]
    Cv = np.linalg.lstsq(Ce, Cr[:,1])[0]
    
    calibmat = np.array([Ch,Cv])
    
    # Plot nr 2: Pre-recorded calibration data, in deg -----------------------
    # Use the calibration matrix for calibrating the pupil position
    eyeRot = arctan(calibmat.dot(append(xy, ones((xy.shape[0], 1)), 1).T)).T
    normEyeRot = sqrt(apply_over_axes(sum, eyeRot**2, 1))
    eyeRotVec = concatenate((zeros((eyeRot.shape[0], 1)), eyeRot*tile(tan(0.5*normEyeRot)/normEyeRot, (1, 2))), 1)
    eyeRotVec = eyeRotVec[:, [0, 2, 1]]
    eyeRotDeg = eyeRot * 180.0 / pi   # Transform to angular degrees
    
    # Make the plot
    ax1 = plt.subplot(2,2,2)
    plt.plot(t, eyeRotDeg)
    plt.ylabel('Eye Position [deg]')
    plt.xlabel('Time [s]')

    # Plot Nr. 4: Eye velocity, of pre-recorded calibration data ----------------------------
    
    # compute angular velocities; sampling rate was 220 Hz
    rdot = diff3(eyeRotVec) * SamplingRate
    w_eye = 2*(rdot + cross(eyeRotVec, rdot))/tile(1 + apply_over_axes(sum, eyeRotVec**2, 1), (1, 3))
    w_eyeDeg = w_eye * 180.0 / pi

    # Plot the velocties
    plt.subplot(2,2,4, sharex=ax1)
    plt.hold(True)
    plt.plot(t, w_eyeDeg[:,2], label='hor')
    plt.plot(t, w_eyeDeg[:,1], label='ver')
    plt.plot(t, w_eyeDeg[:,0], label='tor')
    plt.legend()
    plt.ylabel('Eye Velocity [deg/s]')
    plt.xlabel('Time [s]')
    plt.ylim(-400,400)
    plt.hold(False)
    
    plt.show()

if __name__ == '__main__':
    ''' Main function '''

    # Get the data
    inDir = '.'
    # inFile = 'VIDEO1 LEFT POSTERIOR CANAL BPV.wmv'
    inFile = 'p00006s001e010c001t002.avi'
    myfile = os.path.join(inDir, inFile)
    
    # Open the file, get some basic info
    fh = cv2.VideoCapture(myfile)
    numFrames = fh.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = fh.get(cv2.cv.CV_CAP_PROP_FPS)
    
    # Generate the video-window
    success, frame = fh.read()
    cv2.namedWindow('video')
    
    # Just for development, so I don't have to wait so long:
    numFrames = min(1000, numFrames)
    
    # Allocate the memory
    centerPupil = np.nan*np.ones((numFrames,2))
    ii = 0
    threshold = 40
    
    while success and cv2.waitKey(1) and ii < numFrames:
        sys.stdout.write("\rPercent done: %d  " % (ii*100/numFrames))
        
        # Find the pupil center
        centerPupil[ii], threshold = findCenter(frame, threshold)
        
        # Show the current frame, with the center indicated
        cv2.circle(frame, (int(centerPupil[ii,0]), int(centerPupil[ii,1])), 10, (0, 0,255))
        cv2.imshow('video', frame)
        #cv2.waitKey(10)
        
        # Proceed to the next frame
        ii += 1
        success, frame = fh.read()
        
    # Close the CV-window    
    cv2.destroyAllWindows()
    fh.release()

    
    # Eliminate un-detected pupil locations
    centerPupil[centerPupil==0] = np.nan

    # Save the data to an outfile
    outFile = 'pupilcenter.txt'
    np.savetxt(outFile, centerPupil)
    print('\nOutput written to {0}'.format(outFile))

    # Make a plot of the output
    time = np.arange(numFrames)/float(fps)
    plotAndCalibration(time, centerPupil)
    
