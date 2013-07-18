'''Simple program for finding a pupil.

- Finds a pupil center
- indicates the center
- generates an outplot
- saves the data

Author: Thomas Haslwanter
Date:   July 2013
Ver:    1.0

'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

def adaptThreshold(img, thresh):
    ''' Adapt threshold '''
    h,w = img.shape
    lower = h*w/50
    upper = h*w/4

    bw = img < thresh

    while np.sum(bw) < lower:
        thresh += 5
        bw  = img<thresh
    while np.sum(bw) < upper:
        thresh += 5
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
    
    # Fill the corners
    # Note that the mask has to be 2 pixels larger than the image!
    mask = np.zeros( (h+2,w+2), dtype=np.uint8)
    cv2.floodFill(im_bw, mask, (1,479), 255)
    
    # After the floodfill operation, the "mask" indicates which areas have
    # been filled. It therefore has to be re-set for the next filling task.
    mask = np.zeros( (h+2,w+2), dtype=np.uint8)
    cv2.floodFill(im_bw, mask, (639,479), 255)
    
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
        
if __name__ == '__main__':
    ''' Main function '''

    # Get the data
    inDir = '.'
    inFile = 'VIDEO1 LEFT POSTERIOR CANAL BPV.wmv'
    myfile = os.path.join(inDir, inFile)
    
    # Open the file, get some basic info
    fh = cv2.VideoCapture(myfile)
    numFrames = fh.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = fh.get(cv2.cv.CV_CAP_PROP_FPS)
    
    # Generate the video-window
    success, frame = fh.read()
    cv2.namedWindow('video')
    
    # Just for development, so I don't have to wait so long:
    numFrames = min(10000, numFrames)
    
    # Allocate the memory
    centerPupil = np.nan*np.ones((numFrames,2))
    ii = 0
    threshold = 70
    
    while success and cv2.waitKey(1) and ii < numFrames:
        sys.stdout.write("\rPercent done: %d  " % (ii*100/numFrames))
        
        # Find the pupil center
        centerPupil[ii], threshold = findCenter(frame, threshold)
        
        # Show the current frame, with the center indicated
        cv2.circle(frame, (int(centerPupil[ii,0]), int(centerPupil[ii,1])), 10, (0, 0,255))
        cv2.imshow('video', frame)
        
        # Proceed to the next frame
        ii += 1
        success, frame = fh.read()
        
    # Close the CV-window    
    cv2.destroyAllWindows()
    
    # Eliminate un-detected pupil locations
    centerPupil[centerPupil==0] = np.nan

    # Save the data to an outfile
    outFile = 'pupilcenter.txt'
    np.savetxt(outFile, centerPupil)
    print('\nOutput written to {0}'.format(outFile))

    # Make a plot of the center position
    time = np.arange(numFrames)/float(fps)
    plt.plot(time, centerPupil[:,0], label='horizontal')
    plt.hold(True)
    plt.plot(time, centerPupil[:,1], label='vertical')
    plt.legend()
    plt.show()
