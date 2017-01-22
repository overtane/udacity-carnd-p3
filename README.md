## Udacity Self-Driving Car Nanodegree

# Project 3 - Behavioral Cloning

<img src="images/behavioral_cloning.jpg" width="480" alt="Crash course" />

This is ...

## 1. Getting Data

I started with the Udacity data set. The problem with this data is that it has excess zero-angle samples and a lot of jumps
to zero (actually gaps in data). This is probably because of the recording method, which does not correcpond to real driving conditions.

Because of that, I soon ended up recording data myself. Luckily I had access to PS3 controller, which I could pair with my MacBook. 
First I recorded two laps, one each direction. The final data-set consists of images of four full laps, two clockwise and
two counter-clockwise. This gives rather even distribution of data.  

Alltogether the dataset consisted of 12,384 images. There was still many zero-angle images and images very close to zero-angle.
The histogram below illustrates the distribution of steering angles in the data.

<img src="images/angle_distribution.png" width="480" alt="Angle distribution" />

The amount of left and right turns vs. straight driving samples

* Zeroes [-0.01,0.01]: 3400
* Left [-1.0, 0.01]: 4370
* Right [0.01, 1.0]: 4614

The total sum on steering angles tells that there are a little bit more or steeper right turns than left turns.

* Sum left angles: -363.93844251
* Sum right angles: 404.24960682


## 2. Data Pre-Processing

The motivation for collecting a large data set was, that one could avoid data augmentation. Although very useful in a general case,
augmentation would not be necessary in this special case if the data covered large enough set of situations. 

First pre-processing was done to image list (driving_log.csv). All images that had larger steering angle than abs(0.5) were remove.
There was only a few of those. These situations were human errors in operating joystick.

During the work, several pre-processing techniques were considered:
* Brightness equalization. This was done in YCrCB space (later also HSV). Equalization increases contrast of the picture
radically and makes the road area very distinct and easily observable for a human eye.
* Using only lumination channel of the image. Monochromatic image is enough in this case.
* Random brightness variations.
* Using HSV image instead of RGB
* Cropping: crop area was first 90x270 pixels area. Upper left corner where the crop was taken was (25,45). This removed quite 
of amount of sky and forest above the horizon, and the lowest part of the image.   
* Shifting image horizontally and adjusting steering angle accordingly. This was implemented by varying the crop place on the original image. 
This was not used in the final model.
* Resizing: the very first model used 16x48 images resized from 90x270 crops.


The final version uses following image pre-processing. All this is done 'on the fly' via data generator (see below):
* Cropping: final crops have size of 90x320 and taken from upper left corner (50,0). Full width allows the model to use whole image width.
* The crop of 90x320 are resized to 66x200, which is the original NVIDIA size. 
* Converting RGB to HSV color space, and redusing image to saturation channel only (monochromatic). Using HSV was empirically observed to be best 
approach and saturation channel alone gave equally good results.
* Normalizing values to [-0.5,0.5]. A good practice to the mean of training data close to zero. This could have been done in Keras model,
but because I was not using GPU, I'd like to do it separately and have better control to what is happening (for example can preview the images 
that come out of the preprocessing.)


Sample camera picture and the corresponding generated data frame.

<img src="images/preprocess.png" width="480" alt="Angle distribution" />


## 3. The Model




____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    624         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 31, 98, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 14, 47, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 22, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 20, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       activation_4[0][0]               
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 18, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        activation_6[0][0]               
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         activation_7[0][0]               
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
y_pred (Dense)                   (None, 1)             11          activation_8[0][0]               
====================================================================================================
Total params: 251,019
Trainable params: 251,019
Non-trainable params: 0

## 4. The Generator


## 5. Training


## 6. Validating and Testing
...
