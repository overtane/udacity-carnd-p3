## Udacity Self-Driving Car Nanodegree

# Project 3 - Behavioral Cloning

<img src="images/behavioral_cloning.jpg" width="480" alt="Crash course" />

This is ...

## 1. Getting Data

I started with the Udacity data set. The problem with this data is that 

Finally I ended up recording the data myself. Luckily I had access to PS3 controller, which I could pair with my MacBook. 
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


<img src="images/preprocess.png" width="480" alt="Angle distribution" />


## 3. The Model


## 4. The Generator


## 5. Training


## 6. Validating and Testing
...
