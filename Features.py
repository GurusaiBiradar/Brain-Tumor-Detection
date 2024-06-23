import matplotlib.pyplot as plt
from scipy import stats
import math
import matplotlib.image as mpimg
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from random import randint
from skimage import feature
import numpy as np
import numpy as np
from scipy import stats
PATCH_SIZE = 20
training_length = 500
from skimage.color import rgb2gray
#angles = [0.0, 45.0, 90.0 , 135.0]
#angles = [float(x) for x in angles]
#angles = [math.radians(x) for x in angles]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/2]
np.set_printoptions(precision=10)

def reshape_list(arr):
	return np.reshape(arr, arr.size)

def feature_glcp(I):
    
    x_len, y_len= I.shape
    x_axes = []
    y_axes= []
    feature = []
    temp = []
    for data in range(0, training_length):
        x_axes.append(randint(0,x_len-1));
        y_axes.append(randint(0, y_len-1));

    for pixels in range(0,training_length):
        vertil = y_axes[pixels]
        horiz  = x_axes[pixels]
            
    temp = np.array([])
    
    for h in angles:
            square = I[vertil:vertil + PATCH_SIZE,horiz:horiz + PATCH_SIZE]
            #print(square.shape)
            if square.shape[0]!=0 and square.shape[1]!=0 :
                glcm = greycomatrix(square, [1 ,2, 3], [h], 256, symmetric=True, normed = True)
                dissim = (greycoprops(glcm, 'dissimilarity'))
                correl = (greycoprops(glcm, 'correlation'))
                energy = (greycoprops(glcm, 'energy'))
                contrast= (greycoprops(glcm, 'contrast'))
                homogen= (greycoprops(glcm, 'homogeneity'))
                asm =(greycoprops(glcm, 'ASM'))
                glcm = glcm.flatten()
                statistics = stats.describe(glcm)
                temp1 = [statistics.mean ,statistics.variance ,statistics.skewness ,statistics.kurtosis]
                check = np.append(dissim, correl)
                check = np.append(check, energy)
                check = np.append(check,contrast) 
                check = np.append(check, homogen)
                check = np.append(check, asm)
                check = np.append(check,temp1)
                temp = np.append(temp,check)
            else:
                temp1 = [0 ,0 ,0 ,0]
                check = np.append(0, 0)
                check = np.append(check, 0)
                check = np.append(check,0) 
                check = np.append(check, 0)
                check = np.append(check, 0)
                check = np.append(check,0)
                temp = np.append(temp,0)
            len(temp)
    return temp

def describe(image,radius,numPoints):
        eps=1e-7
        lbp = feature.local_binary_pattern(image,numPoints,radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, numPoints + 3),
            range=(0, self.numPoints + 2))
 
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
        return hist


def fft_feature(file):
        features = ImageReader.read_from_single_file(file)
        xc=features.moment(output_type="pandas")
        fea=np.array([np.array(xc['mean']),np.array(xc['median']),np.array(xc['var']),np.array(xc['skew']),np.array(xc['kurtosis'])])
        fea=fea.reshape(1,15)
        return fea
        
