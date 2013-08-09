'''Simple program for calibrating eye positions.

Author: Erich Schneider and Thomas Haslwanter
Date:   August 2013
Ver:    1.0

'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

from numpy import *
from scipy.stats import *
from scipy.io import loadmat, savemat
from scipy.cluster.vq import kmeans
from scipy import signal


''' 3-point differentiation
'''
def diff3(x):
    dxdt = zeros(x.shape)
    dxdt[1:-1, :] = 0.5*(x[2:, :] - x[:-2, :])
    dxdt[[0, -1], :] = dxdt[[1, -2], :]
    return dxdt


def heatMap (xy):
    # First calculate the 2D histogram
    hist2d, xedges, yedges = np.histogram2d(xy[:,1], xy[:,0], bins=(100, 100))
    # Then blur the histogram
    filtx = signal.gaussian(15, 15/6.0)
    filtx = filtx/sum(filtx)
    filty = filtx
    hist2d = signal.sepfir2d(hist2d, filtx, filty)

    # Calculate the range / extent    
    extent = array([amin(xy[:,0]), amax(xy[:,0]), amin(xy[:,1]), amax(xy[:,1])])
    return (hist2d, extent)


if __name__ == '__main__':
    ''' Main function '''

    # Open the file, get some basic info
    fh = cv2.VideoCapture('p00006s001e010c001t002.avi')
    
    plt.figure(1)    
    # Get frames at different pupil positions
    for ii in range(40):
        success, frame = fh.read()        
    plt.subplot(3,3,5)
    plt.imshow(frame[...,0])
    plt.gray()
    for ii in range(90):
        success, frame = fh.read()        
    plt.subplot(3,3,6)
    plt.imshow(frame[...,0])
    plt.gray()
    for ii in range(50):
        success, frame = fh.read()        
    plt.subplot(3,3,2)
    plt.imshow(frame[...,0])
    plt.gray()
    for ii in range(50):
        success, frame = fh.read()        
    plt.subplot(3,3,4)
    plt.imshow(frame[...,0])
    plt.gray()
    for ii in range(630):
        success, frame = fh.read()        
    plt.subplot(3,3,8)
    plt.imshow(frame[...,0])
    plt.gray()

    plt.show()

    fh.release()

    plt.figure(2)   
    # Load the pupil center positions
    # The original sampling rate is 220 Hz

    xy = np.loadtxt('pupilposition.txt')
    
    # Remove values that are not a number (gaps, blinks)
    xy = xy[xy[:,0]==xy[:,0],:] #isnan
    
    # The original frame rate was 220 Hz
    SamplingRate = float(220)
    
    # Cluster analysis of the fixations
    ntarget = 5;
    calibtarget = 8.5; # Laser dots are 8.5 deg apart
    clust, distortion = kmeans (xy, ntarget)
    
    # Define fixation directions: center, up, down, right, left
    Cr0 = array(
            [[0, 0],
             [0,-1],
             [0, 1],
             [-1,0],
             [1, 0]])
       
    imax = np.nanargmax(clust, axis=0)
    imin = np.nanargmin(clust, axis=0)
    # Indices of secondary fixations
    # Reordering: up, down, right, left
    iorder = np.array([imax[1], imin[1], imin[0], imax[0]])
    # Index of primary fixation
    imid = setdiff1d(arange(ntarget),iorder)
    # Add center to up, down, right, left
    ic0 = concatenate((imid,iorder))
    # Reorder the kmeans clusters: center, up, ...
    C = clust[ic0,:]
    
    # Multiply fixation directions with real calibration size (8.5 deg)    
    Cr_deg = -Cr0*calibtarget;    
    # Projection into the image plane
    Cr = tan(-Cr_deg * pi/180)
    
    # Extend clusters by ones for the intercept 
    Ce = concatenate((C, ones((C.shape[0],1))), axis=1)
    # Linear fit; In Matlab it's Cr.T / Ce.T
    # Determine gain and offset for a screen distance of one
    # and for aligning the eye orientation angles with the coordinate system
    # see Schreiber and Haslwanter (2004)
    Ch = np.linalg.lstsq(Ce, Cr[:,0])[0]
    Cv = np.linalg.lstsq(Ce, Cr[:,1])[0]
    
    # Assemble corresponding calibration matrix    
    calibmat = np.array([Ch,Cv])
    
    # Time vector
    t = np.arange(xy.shape[0])/SamplingRate
    
    # Use the calibration matrix for calibrating the pupil position
    # Calculate 3D eye orientation in rad assuming zero Listing's torsion 
    eyeRot = arctan(calibmat.dot(append(xy, ones((xy.shape[0], 1)), 1).T)).T
    # Vector length of eye rotation    
    normEyeRot = sqrt(apply_over_axes(sum, eyeRot**2, 1))
    # Calculate 3D eye rotation vector according to Haslwanter (1995), Haustein (1989)
    eyeRotVec = concatenate((zeros((eyeRot.shape[0], 1)), eyeRot*tile(tan(0.5*normEyeRot)/normEyeRot, (1, 2))), 1)
    # Make it a right hand coordinate system
    # x = Torsion, y = Vertical, z = Horizontal
    # Positive is clockwise, left, and downwards    
    eyeRotVec = eyeRotVec[:, [0, 2, 1]]
    
    # compute angular velocities for rotation vectors
    # First, calculate the first derivative of rotation vectors
    # Note that this is not velocity; frame rate was 220 Hz
    rdot = diff3(eyeRotVec) * SamplingRate
    # Compute anguler velocity in rad/sec according to Haslwanter (1995)
    w_eye = 2*(rdot + cross(eyeRotVec, rdot))/tile(1 + apply_over_axes(sum, eyeRotVec**2, 1), (1, 3))

    # Transform to angular degrees    
    eyeRotDeg = eyeRot * 180.0 / pi
    w_eyeDeg = w_eye * 180.0 / pi
    # Ct=arctan((calibmat*Ce.take([0,1],axis=1).T).T)*180/pi


    # Now to something completely different:
    # A heat map; This is just for fun
    heatmap, ex = heatMap (xy)
    
    plt.subplot(3,2,5)
    plt.cla()
    plt.hold(True)
    plt.imshow(heatmap, origin='lower', extent=ex, cmap='afmhot')
    plt.contour(heatmap, origin='lower', extent=ex)
    plt.plot(xy[:,0], xy[:,1], '0.8', C[:,0], C[:,1], 'ro');
    plt.ylabel('Vertical [Pixel]')
    plt.xlabel('Horizontal [Pixel]')
    plt.hold(False)
    
    plt.subplot(3,2,6)
    plt.cla()
    plt.hold(True)
    plt.plot(eyeRotDeg[:,0], eyeRotDeg[:,1], '0.8');
    plt.ylabel('Vertical [deg]')
    plt.xlabel('Horizontal [deg]')
    plt.hold(False)

    ax1 = plt.subplot(3,1,1)
    plt.cla()
    plt.plot(t, eyeRotDeg)
    plt.ylabel('Eye Position [deg]')
    plt.xlabel('Time [s]')

    plt.subplot(3,1,2, sharex=ax1)
    plt.cla()
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




    
