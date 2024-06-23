import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt

def segmentation(img):
    #img = image_example#cv2.imread(im)
    #cv2.waitKey(0)
    median = cv2.medianBlur(img,5) # Apply Median filter
    # =============================================================================
    #      img = cv2.imread(org_imgName,-1)

    Z = median.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    K = 8
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,flags)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    kmeans_img = res.reshape((img.shape))



    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    hsv = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2HSV)# convert from BGR to HSV color space
    cv2.namedWindow("Kmean_img1", cv2.WINDOW_NORMAL)
    cv2.imshow('Kmean_img1',hsv)

    h, s, v = cv2.split(hsv)  # split on 3 different channels
    #apply CLAHE to the L-channel
    h1 = clahe.apply(h)
    s1 = clahe.apply(s)
    v1 = clahe.apply(v)

    lab = cv2.merge((h1,s1,v1))  # merge channels



    Enhance_img= cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR


    # =============================================================================
    #    making the mask for grabcut
    # =============================================================================
    hsv = cv2.cvtColor(Enhance_img, cv2.COLOR_BGR2HSV)    
    lower_green = np.array([50,100,100])
    upper_green = np.array([100,255,255])
    mask_g = cv2.inRange(hsv, lower_green, upper_green)

    ret,inv_mask = cv2.threshold(mask_g,127,255,cv2.THRESH_BINARY_INV)

    res = cv2.bitwise_and(img, img, mask= inv_mask)
    return res
