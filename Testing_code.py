import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#from google.colab import files
import os
import zipfile

from glob import glob
from PIL import Image as pil_image
from matplotlib.pyplot import imshow, imsave
from IPython.display import Image as Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from color_segment import segmentation


import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
from Features import feature_glcp
from skimage.color import rgb2gray

from tkinter import filedialog
filename = filedialog.askopenfilename(title='open')

image_example = np.asarray(pil_image.open(filename))
cv2.imshow('Original Image',image_example)
image_example=cv2.resize(image_example,(256,256))
# =============================================================================
# Get the image
# =============================================================================

img = image_example#cv2.imread(im)
#cv2.waitKey(0)
median = cv2.medianBlur(img,5) # Apply Median filter
# =============================================================================
#      img = cv2.imread(org_imgName,-1)
if len(median.shape)==3:
    Z = median.reshape((-1,3))
    hsv1=median
else:
    
    hsv1=np.zeros((median.shape[0],median.shape[1],3))
    hsv1[:,:,0] = median
    hsv1[:,:,1] = median
    hsv1[:,:,2] = median
    Z = hsv1.reshape((-1,3))
        
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
kmeans_img = res.reshape((hsv1.shape))

#cv2.imshow('Color Processed',kmeans_img)

# =============================================================================
# =============================================================================
#    Adaptive histogram equalization  
# =============================================================================
clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
if len(kmeans_img.shape)==3:
    hsv = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2HSV)# convert from BGR to HSV color space
else:
    hsv=np.zeros((kmeans_img.shape[0],kmeans_img.shape[1],3))
    hsv[:,:,0] = kmeans_img
    hsv[:,:,1] = kmeans_img
    hsv[:,:,2] = kmeans_img
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)# convert from BGR to HSV color space

cv2.imshow('Segmented Image',hsv)

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
ret1,output = cv2.threshold(kmeans_img ,160,255,cv2.THRESH_BINARY_INV)


for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        for k in range(output.shape[2]):

            if output[i,j,k]==255:
                output[i,j,k]=0
            else:
                output[i,j,k]=255
            
    


cv2.imshow('Segmented Image1',output)
res = cv2.bitwise_and(img, img, mask= inv_mask)



xc=segmentation(cv2.imread(filename))
if len(xc.shape)==3:
    grayscale = rgb2gray(xc)*255
else:
    grayscale = (xc)*255
grayscale=grayscale.astype(int)
ln=0

while (ln<50):
    xc=feature_glcp(grayscale)
    ln=len(xc)
        

X=xc#np.transpose(input1[:,1:])
#X=input1[:,1:]


from keras.models import load_model


#model = load_model('MLBPNN_model_.h5')
import pickle
model=pickle.load( open('MLBPNN_model_.h5', "rb"))
pred=model.predict(X.reshape(1,88))
                           
if np.argmax(pred)==0:
    print('Given Image is Normal ')
else:

    if np.count_nonzero(output)<10000:
        print('Given Image is Dieseased and Tumour level is 1 ')

    else:
        print('Given Image is Dieseased and Tumour level is 2 ')

