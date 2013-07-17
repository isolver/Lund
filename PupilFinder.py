'''First version of finding a pupil.

Not fast, not pretty, but gets the job done.
- Finds a pupil center
- indicates the center
- generates an outplot
- saves the data

Author: Thomas Haslwanter
Date:   July 2013
Ver:    0.1

'''

from FindPupil import *
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

def findCenter(frame):    
    '''Take a frame, and find the pupil center'''
    img = frame[...,0]
    h,w = img.shape
    
    # Set an (adaptive) threshold. Note that the adaptation goes only into one direction!
    threshold = 70
    bw  = img<threshold
    while np.sum(bw) < 10000:
        threshold += 5
        bw  = img<threshold
        
    # Flip b/w, and convert to uint8
    im_bw = np.array(~bw, dtype=np.uint8)*255
    
    # Fill the corners
    mask = np.zeros( (h+2,w+2), dtype=np.uint8)
    cv2.floodFill(im_bw, mask, (1,479), 255)
    
    mask = np.zeros( (h+2,w+2), dtype=np.uint8)
    cv2.floodFill(im_bw, mask, (639,479), 255)
    
    # Fill the holes
    wHoles = im_bw.copy()
    # Note that the mask has to be 2 pixels larger than the image!
    mask = np.zeros( (h+2,w+2), dtype=np.uint8)
    cv2.floodFill(wHoles, mask, (1,1), 0)
    im_bw[wHoles==255] = 0
    
    # Close the image, to remove the eye-brows    
    strEl = makeCircle(25)
    closed = close(im_bw, strEl)
    
    # find the edge
    thrs1 = 1000
    thrs2 = 1000
    edge = cv2.Canny(closed, thrs1, thrs2, apertureSize=5)
    
    # find the center of the edge
    edge_y, edge_x = np.where(edge==255)
    center = np.array([np.mean(edge_x), np.mean(edge_y)])
    if np.any(np.isnan(center)):
        center = np.zeros(2)
        
    return center
        
if __name__ == '__main__':
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
    maxLength = 100
    
    # Allocate the memory
    centerPupil = np.zeros((maxLength,2))
    ii = 0
    
    while success and cv2.waitKey(1) and ii < maxLength:
        sys.stdout.write("\rPercent done: %d  " % (ii*100/maxLength))
        
        # Find the pupil center
        centerPupil[ii] = findCenter(frame)
        
        # Show the current frame, with the center indicated
        cv2.circle(frame, (int(centerPupil[ii,0]), int(centerPupil[ii,1])), 10, (0, 0,255))
        cv2.imshow('video', frame)
        
        # Proceed to the next frame
        ii += 1
        success, frame = fh.read()
        
    # Close the CV-window    
    cv2.destroyAllWindows()
    
    # Make a plot of the center position
    time = np.arange(maxLength)/float(fps)
    plt.plot(time, centerPupil[:,0], label='horizontal')
    plt.hold(True)
    plt.plot(time, centerPupil[:,1], label='vertical')
    plt.legend()
    plt.show()
    
    # Save the data to an outfile
    outFile = 'pupilcenter.txt'
    np.savetxt(outFile, centerPupil)
    print('Output written to {0}'.format(outFile))
    raw_input('Done')