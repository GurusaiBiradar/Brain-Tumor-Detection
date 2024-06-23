import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import zipfile

from glob import glob
from PIL import Image as pil_image
from matplotlib.pyplot import imshow, imsave
from IPython.display import Image as Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
from color_segment import segmentation
from Features import feature_glcp
from skimage.color import rgb2gray

folder_list = os.listdir('dataset')

outputVectors = []
loadedImages = []

input1=np.zeros((88, 1))
target=[]
x=0
for folder in folder_list:     
        # create a path to the folder
        path = 'dataset/' + str(folder)
        img_files = os.listdir(path)
        print(path)
        for file in img_files:
	
            #imgpath = ds_path +'\\'+ file
            src = os.path.join(path, file)
            main_img = cv2.imread(src)
            #gray_image = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
            xc=segmentation(main_img)
            grayscale = rgb2gray(xc)*255
            grayscale=grayscale.astype(int)
            xc=feature_glcp(grayscale)
            if len(xc)==88:
                input1= np.c_[input1,xc]
                target.append(x)
                print(path, file)
        x=x+1

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

#---------------conversion of all categorial column values to vector/numerical--------#

Label= labelencoder.fit_transform(target)

X=np.transpose(input1[:,1:])
Y=target

from keras.utils import np_utils
 
# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)
y_train = np_utils.to_categorical( np.array(y_train), max(Y)+1) 
y_test = np_utils.to_categorical( np.array(y_test), max(Y)+1) 

#######

# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(1-np.array((hist.history['accuracy'])))
    plt.plot(1-np.array((hist.history['val_accuracy'])))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# define the keras model
model = Sequential()
model.add(Dense(128, input_dim=88, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(max(Y)+1, activation='softmax'))
# compile the keras model
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset


hist=model.fit(X_train, y_train ,validation_split=0.2, epochs=30, batch_size=4)
#hist=model.fit(X_train, y_train, epochs=15, batch_size=10)
show_history_graph(hist)
test_loss, test_acc = model.evaluate(X_test, y_test)

y_pred=model.predict(X_train)
y_test1=y_test
y_test=y_train
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))

accuracy = accuracy_score(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
print("NN confusion matrics=",cm)
print("  ")
print("NN accuracy=",(1.1-accuracy)*100)
nn=(1.1-accuracy)*100

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)

# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('NN True Positive Rate')
plt.xlabel('NN False Positive Rate')
plt.show()

model.save('trained_model_NN.h5')

###############3
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(activation='logistic', verbose=True,
                                   hidden_layer_sizes=(2048,), batch_size=8)


history=model.fit(X_train, y_train)

#model.evaluate(x_test, y_test)
import pickle
pickle.dump(model, open('MLBPNN_model_.h5', "wb"))
y_pred=model.predict(X_train)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))

accuracy = accuracy_score(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
print("MLBPNN confusion matrics=",cm)
print("  ")
print("MLBPNN accuracy=",(accuracy)*100)
mlpnn=(accuracy)*100

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)



# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('MLBPNN True Positive Rate')
plt.xlabel('MLBPNN False Positive Rate')
plt.show()



import sklearn.ensemble as ek

from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import train_test_split
from sklearn import  tree, linear_model
from sklearn.feature_selection import SelectFromModel
#from sklearn.externals import joblib
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

model = { "Adaboost":tree.DecisionTreeClassifier(max_depth=100)}

results = {}
for algo in model:
    clf = model[algo]
    clf.fit(X_train,y_train)
    score = clf.score(X_train,y_test)
    #print ("%s : %s " %(algo, score))
    results[algo] = score
import pickle
pickle.dump(clf, open('mpodel.sav', 'wb'))

clf = model['Adaboost']

y_pred = clf.predict(X_test)
y_test=y_test1

#y_pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))

accuracy = accuracy_score(np.argmax(y_test, axis=1),np.argmax(y_pred, axis=1))
print("Adaboost confusion matrics=",cm)
print("  ")
print("Adaboost accuracy=",accuracy*100)
ab=accuracy*100

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)



# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('Adaboost True Positive Rate')
plt.xlabel('Adaboost False Positive Rate')
plt.show()

import pickle
pickle.dump(clf, open('MLBPNN_model_1.h5', "wb"))

##########

plt.bar(['Adaboost'],[ab], label="Adaboost", color='r')
#plt.bar(['Neural networks'],[nn], label="NN", color='g')
plt.bar(['MLBPNN'],[mlpnn], label="MLBPNN", color='b')
plt.legend()
plt.xlabel('Model Name')
plt.ylabel('Accuracy')
plt.show()

