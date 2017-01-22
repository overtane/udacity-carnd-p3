import pandas as pd
import numpy as np
import math
import cv2

def load_data_frame():
    project_path = '/Users/olli/Udacity/SDC/P3'
    data_path = project_path + '/simulator/data'
    log_name = data_path + '/driving_log.csv'
    df = pd.read_csv(log_name)
    # drop outliers
    df0 = df.query('steering > -0.5 and steering < 0.5')
    df = df0.reset_index(drop=True)
    return df

#
# Image read and pre-processing here
# - data generator uses these routines
#

def get_image_and_angle(dfx, data_path, index, pos=1):
    """
    Read image data and get the steering angle corresponding the image 
    Angle is corrected, if the camera is left or right
    """
    camera = ['left', 'center', 'right']

    data_file = data_path + '/' + dfx.loc[index,camera[pos]].strip()
    img = mpimg.imread(data_file)

    angle = dfx.loc[index,'steering']
    if pos == 0 or pos == 2:
        camera_corr = abs(angle) * np.random.uniform(2.0,4.0)
        angle = angle +  (pos * -camera_corr + camera_corr)
        angle = min(angle, 0.8) # clip too large values
    return img, angle

def normalize_image(img):
    return img.astype('float') / 255.0 - 0.5 # normalize to [-0.5,0.5]

def get_cropped_image(dfx, data_path, index, pos=None):
    """
    Return a cropped camera image (left, right or center)
    If pos (camera) == None, return a random crop from a random camera
    
    """
    if (pos == None):
        pos = np.random.randint(2)*2
        shift = np.random.randint(-25,25) # shift value, not used in this version
    else:
        shift = 0
        
    img, angle = get_image_and_angle(dfx, data_path, index=index, pos=pos)

    # flip image randomly in order to avoid left/right bias in a batch
    if np.random.randint(2):
        img = cv2.flip(img,1)
        angle = -angle
        
    img2 = img[50:140,:] # crop
    img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV) # change color space
    img3 = cv2.resize(img3, (200,66), interpolation=cv2.INTER_AREA) #resize

    return normalize_image(img3)[:,:,1].reshape(66,200,1), angle2


#
# Data generator
#
#
from sklearn.utils import shuffle

def get_image(dfx, data_path, index, pos=None, threshold=0.0):
   
    img, angle = get_cropped_image(dfx, data_path, index=index, pos=pos)
    return img, angle
        
def index_generator(dfx, batch_size):
    
    data_set_size = len(dfx)
    i = 0 # data
    j = 0 # batch
    iarray = np.arange(len(dfx)) #index array
    iarray = shuffle(iarray)
    
    while True:
        
        if i >= data_set_size:
            # shuffle index array every time we have used whole set of original images
            iarray = shuffle(iarray) 
            i = 0
        if j >= batch_size:
             j = 0

        yield iarray[i], j

        i = i + 1
        j = j + 1

        
def data_generator(dfx, batch_size, val=False, train=0):
    
    igen = index_generator(dfx, batch_size) 
    X_batch = np.zeros( (batch_size, 66, 200, 1), dtype='float')
    y_batch = np.zeros( batch_size, dtype='float')
    
    if val == True:
        pos = None
    elif train == 1:
        # randomly take either left or right camera to the batch
        pos = np.random.randint(2) * 2
    else:
        # train only with center camera - used in ‘main’ training passes
        pos = 1
    
    while True:
        data_index, batch_index = next(igen)

        if val == False and train == 1:
            pos = np.random.randint(2) * 2
        X, y = get_image(dfx, data_path, index=data_index, pos=pos)
            
        X_batch[batch_index] = X
        y_batch[batch_index] = y

        if batch_index == batch_size - 1:
            yield X_batch, y_batch  

    
#
# 
# The Model
#
#

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam

def nvidia_model():
    """
    Implementation of the 'NVIDIA CNN' from 'End to End Learning for Self-Driving Cars'
    (https://arxiv.org/pdf/1604.07316v1.pdf)
    
    Normalization of the input, part of the original architecture, is done outside of this function
        
    """
    model = Sequential()
    
    #model.add(Convolution2D(3, 1, 1, input_shape=(90,270,3)))
    #model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)))
    
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), input_shape=(66,200,1)))

    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(100))
    #model.add(Dense(100, W_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(50))
    #model.add(Dense(50, W_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(10))
    #model.add(Dense(10, W_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.50))
    #model.add(Dense(1, name='y_pred', W_regularizer=l2(0.01)))
    model.add(Dense(1, name='y_pred'))
    return model

model = nvidia_model()
modelname = 'nvidia'
model.summary()


from keras.models import model_from_json

def model_save(path, name, epoch):
    """
    Save model file and weights after each epoch
    (Actually model is the same every time, but this is easier way to integrate to drive.py)
    """
    model_path = path + '/model'
    filename = model_path + '/' + name + '_' + str(epoch)
    model_file =   filename + '.json'
    weights_file = filename + '.h5'

    json_string = model.to_json()
    with open(model_file, 'w') as jfile:
        jfile.write(json_string)
    jfile.close()

    model.save_weights(weights_file)
    
def model_restore(path, name, epoch):
    """
    Restore model and weights
    """
    model_path = path + '/model'
    filename = model_path + '/' + name + '_' + str(epoch)
    model_file =   filename + '.json'
    weights_file = filename + '.h5'
    
    with open(model_file, 'r') as jfile:
        json_string = jfile.read()
    jfile.close()
    
    model = model_from_json(json_string)
    model.load_weights(weights_file, by_name=False)
    
    return model


from keras.callbacks import Callback

class LossHistory(Callback):
    name = "" # model name
    path = "" # project path

    def __init__(self, path, name, epoch):
        self.path = path
        self.name = name
        self.epoch = epoch
    def on_epoch_end(self, epoch, logs={}):
        model_save(self.path, self.name, self.epoch+epoch)
        #print("Model saved")
        # shuffle input after each epoch
    def on_batch_end(self, batch, logs={}):
        self.batch = 0



def train_model(model, modelname, model_version, project_path):
    “””
    modelname: readable name of the model
    model_version: version number of the first epoch, for saving the model
    projeckt_path: filesystem path for saving the model
    “””

    print ("Training model", modelname)

    model.compile(loss='mse', optimizer=Adam())

    history = LossHistory(project_path, modelname, model_version)

    model.fit_generator(data_generator(df, batch_size=128), 
                              samples_per_epoch = 128*8*12, # 12288 samples 
                              nb_epoch=1, 
                              verbose=1,
                              #validation_data=data_generator(df,batch_size=128, val=True),
                              #nb_val_samples=128*8*3,
                              validation_data=None,
                              callbacks=[history])

    return model

def finetune_model(modelname, model_version, project_path):
    “””
    modelname: readable name of the model
    model_version: version number to restore
    projeckt_path: filesystem path for saving the model
    “””
    model = model_restore(project_path, modelname, model_version)

    model.compile(loss='mse', optimizer=Adam())

    history = LossHistory(project_path, modelname, model_version*10) # child version 

    model.fit_generator(data_generator(df, batch_size=128, val=False, train=1), 
                              samples_per_epoch = 128*8*4, 
                              nb_epoch=1, 
                              verbose=1,
                              #validation_data=data_generator(df,batch_size=128, val=True),
                              #nb_val_samples=128*8*2,
                              validation_data=None,
                              callbacks=[history])
    return model


