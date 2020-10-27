# README for project: comparing_filters

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description:   
- This purpose of this package is to allow a user to compare the effect of different edge detection filters.
- The user selects an image, and the code outputs a montage displaying the effect of each individual filter (listed blow).
  - [A Laplacian filter](https://en.wikipedia.org/wiki/Gaussian_filter#:~:text=In%20electronics%20and%20signal%20processing,as%20it%20has%20infinite%20support) (smooths/blurs image). 
  - [A Sobel-x and Sobel-y filter](https://en.wikipedia.org/wiki/Sobel_operator). 
  - [A Canny filter](https://en.wikipedia.org/wiki/Canny_edge_detector).  
- For each ```.py``` file provided, there is an accompanying ```.ipynb``` file for those who use JupiterLab.

## How to use the ```RUNME.py``` code: 

(1) First, open the RUNME.py file. 

(2) Within the ```RUNME.py``` file, first load in the module functions with the following code: 
```
# Import the necessary packages.
# Function input args: None. 
# Function returns: The file path corresponding to the image selected by the user. 
from comparing_edge_detection_filters import *
```

(3) Then, with the following function...
```
# A function to allow the user to select the image they wish to analyse. 
# Function input args: none. 
# Function output 1: The file path of the image in question. 
file_path = file_selection_dialog()
```
... a GUI will appear (see the image below), within which, the user should select the image they which to filter and segment.

<img src="https://github.com/SamHSoftware/Image-Analysis/blob/main/comparing-edge-detection-filters/img/File%20selection.PNG?raw=true" alt="file selection GUI" width="500"/>

The following image (called original.png) is provided as a test input. Within the image, you can see a colony of humans stem cells.

You can find this image, and others needed for unit testing, within [this folder](https://github.com/SamHSoftware/Image-Analysis/blob/main/comparing-edge-detection-filters/img/File%20selection.PNG).

(4) Upon loading in the image of your choice, it will be gaussian smoothed to remove noise and a selection of filters will then be applied (gaussian, median, maximum and minimum filters) with the following code: 

```
# A function to compare the effects of different image filters.
# Function input arg 1: file_path --> A string containing the file path to the image in question. 
# Function input arg 2: plot_images --> Set to True or False. If True, a montage of filtered images will be displayed in the console. 
# Function input arg 3: save_plot --> Set to True or False. If True, a montage of filtered images will be saved to the same directory as the source image.
# Function input arg 4: kernel_size --> Set to positive integer representing size of gaussian kernel when smoothing image. 
# Function input arg 5: canny_thresh_1 --> Set to positive integer representing first threshold for the canny filter hysteresis procedure.
# Function input arg 6: canny_thresh_2 --> Set to positive integer representing second threshold for the canny filter hysteresis procedure.
# Function output 1: The montage of filtered images. 
montage = compare_edge_detection(file_path, plot_images, save_plot, kernel_size, canny_thresh_1, canny_thresh_2)
```

The code will then output/save the following montage, depending on the input arguments you choose. 

<img src="https://github.com/SamHSoftware/Image-Analysis/blob/main/comparing-edge-detection-filters/img/original_comparing_edge_detection_filters.png?raw=true" alt="montage of filter effects" width="500"/>

If the montage is saved, it will be saved to the same directory as the image which you originally selected. 

## Testing the ```compare_edge_detection()``` function: 

(1) First, open ```test_comparing_edge_detection_filters.py```

(2) Run the following code to import the ```compare_filters()``` function and the other necessary packages. 

```
# Import any necessary packages and modules. 
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np 
import scipy.ndimage
```

(3) Then, run the following function (and it's corresponding code) within the python console such that it may be called and used.
```
# Test the function 'comparing_filters()' against the provided output. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_compare_edge_detection():
  # The function code is here in test_comparing_edge_detection_filters.py.
```

If there are errors, then the code will print assertion errors explaining what went wrong.  

If there were no errors, then a message confirming this will be printed. 
