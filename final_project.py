#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from random import randint
import matplotlib.gridspec as gridspec


# In[4]:


def get_data(dir):
    images = []
    categories = []
    category = 0
    
    for label in os.listdir(dir):
        if label == 'buildings':
            category = 0
        elif label == 'forest':
            category = 1
        elif label == 'glacier':
            category = 2
        elif label == 'mountain':
            category = 3
        elif label == 'sea':
            category = 4
        elif label == 'street':
            category = 5
       
        for image_file in os.listdir(dir+'/'+label):
            image = cv2.imread(dir+ '/' + label+r'/'+image_file)
            image = cv2.resize(image,(150,150))
            images.append(image)
            categories.append(category)
    
    return shuffle(images,categories,random_state = 4560321)

def get_classlabel(class_num):
    categories = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}
    return categories[class_num]


# In[5]:


train_images, train_labels = get_data("seg_train")
test_images, test_labels = get_data("seg_test")

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)


# In[6]:


#Since we are missing a validation set, we combine everything and divide all over again
all_images = np.concatenate((train_images, test_images))
all_labels = np.concatenate((train_labels, test_labels))


# In[7]:


print("Shape of Images:",all_images.shape)
print("Shape of Labels:",all_labels.shape)


# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(all_images, all_labels, train_size = 0.8)


# In[19]:


print("Shape of Train Images:",X_train.shape)
print("Shape of Train Labels:",Y_train.shape)
print("Shape of Validation Images:",X_valid.shape)
print("Shape of Validation Labels:",Y_valid.shape)
print("Shape of Test Images:",X_test.shape)
print("Shape of Test Labels:",Y_test.shape)


# In[20]:


X_train_demo = X_train[:2000]
Y_train_demo = Y_train[:2000]


# In[11]:


from tensorflow.keras.utils import plot_model

model = Models.Sequential()

model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(6,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()


# In[9]:


#trained = model.fit(X_train_demo,Y_train_demo,epochs=5,validation_split=0.30)


# In[12]:


import tensorflow as tf 
trained = tf.keras.models.load_model('my_model.h5')
#model.save('my_model.h5')


# In[13]:


X_test_test = X_test[:400]
Y_test_test = Y_test[:400]

trained.evaluate(X_test_test,Y_test_test, verbose=0)


# In[14]:



X_testest = X_test_test.astype(float)
result = trained.predict(X_testest, verbose=0)
class_result=np.argmax(result,axis=-1)


# In[15]:


print(class_result)


# In[16]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(Y_test_test, class_result)
print(matrix)


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

categories = {'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'}


## confusion matrix from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print_confusion_matrix(matrix, categories, (7,7),(12))
print(accuracy_score(class_result, Y_test_test))

