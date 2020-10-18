# README for project: comparing_filters

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description:   
- This purpose of this package is to allow a user to compare the effect of different filters which tend to smooth/blur images. 
- The user selects an image, and the code outputs a montage displaying the effect of each individual filter (listed blow), alongside the segmentation output (with a threshold determined by the Otsu method). 
  - [A gaussian filter](https://en.wikipedia.org/wiki/Gaussian_filter#:~:text=In%20electronics%20and%20signal%20processing,as%20it%20has%20infinite%20support) (smooths/blurs image). 
  - [A median filter](https://en.wikipedia.org/wiki/Median_filter#:~:text=The%20median%20filter%20is%20a,edge%20detection%20on%20an%20image) (smooths/blurs image). 
  - [A maximum filter](https://reference.wolfram.com/language/ref/MaxFilter.html) (does not smooth/blur image, but yields similar effect).  
  - [A minimum filter](https://reference.wolfram.com/language/ref/MinFilter.html) (does not smooth/blur image, but yields similar effect). 
- For each ```.py``` file provided, there is an accompanying ```.ipynb``` file for those who use JupiterLab.

## How to use the ```RUNME.py``` code: 

(1) First, open the RUNME.py file. 

(2) Within the ```RUNME.py``` file, first load in the module functions with the following code: 
```
# Import functions from the module file. 
from comparing_filters_functions import *
```

(3) Then, with the following function...
```
# Function input args: None. 
# Function returns: The file path corresponding to the image selected by the user. 
file_path = file_selection_dialog()
```
... a GUI will appear (see the image below), within which, the user should select the image they which to filter and segment.

<img src="https://github.com/SamHSoftware/Image-Analysis/blob/main/comparing-filters/img/File%20selection.PNG?raw=true" alt="file selection GUI" width="500"/>

The following image (called original.png) is provided as a test input. Within the image, you can see a colony of humans stem cells.

You can find this image, and others needed for unit testing, within [this folder](https://github.com/SamHSoftware/Image-Analysis/tree/main/comparing-filters/img).

(4) Upon loading in the image of your choice, a selection of filters will be applied (gaussian, median, maximum and minimum filters) with the following code: 

```
# Function input arg 1: The file path of the image which needs to be segmented. 
# Function input arg 2: 'True' to display the figure which compares the filtered/original images to their segmentation outputs. 
# Function input arg 3: 'True' to save the figure which compares the filtered/original images to their segmentation outputs.
# Function input arg 4: The desired colormap to display the images. Must be a string. 
# Function input arg 5: Kernel size for filtering. 
# Function output 1: The figure. 
figure = compare_filters(file_path, plot_images, save_plot, colormap, kernel_size)
```

The code will then output/save the following montage, depending on the input arguments you choose. 

<img src="https://github.com/SamHSoftware/Image-Analysis/blob/main/comparing-filters/img/nuclei_comparing_filters.png?raw=true" alt="montage of filter effects" width="500"/>

If the montage is saved, it will be saved to the same directory as the image which you originally selected. 

## Testing the ```compare_filters()``` function: 

(1) First, open ```test_comparing_filters_functions.py```

(2) Run the following code to import the ```compare_filters()``` function and the other necessary packages. 

```
# Import the module containing the functions we need to unit test. 
import comparing_filters_functions

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
test_compare_filters()
```
If there are errors, then the code will print assertion errors explaining what went wrong.  

If there were no errors, then a message confirming this will be printed. 
