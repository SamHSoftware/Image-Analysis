# README for project: otsu_segmentation

## Author details: 
Name: Sam Huguet  
E-mail: samhuguet1@gmail.com

## Description:   
- This purpose of this package is to allow a user to select an image of their choice and have it segmented.  
- The segmentation threshold is chosen using the Otsu algorithm.  
- The Otsu algorithm examines a data-set, and searches for the threshold that minimises intra-class variance. If you're interested, [read more about it here](https://en.wikipedia.org/wiki/Otsu%27s_method). 

## Loading in the module functions: 

To load in the module functions, use the following code: 
```
# Import functions from the module file. 
from otsu_segmentation_functions import *
```
## Input: 
Upon runnig the following function... 
```
# Function input args: None. 
# Function returns: The file path corresponding to the image selected by the user. 
file_path = file_selection_dialog()
```
... a GUI will appear, whithin which, the user should select the image they which to segment. 

![File selection GUI](https://raw.githubusercontent.com/SamHSoftware/Python/main/Image%20Analysis/otsu_segmentation/img/File%20selection.PNG)

The following image is provided as a test input. Within the image, you can see a colony of humans stem cells.  

You can find this image, and others, wihtin [this folder](https://github.com/SamHSoftware/Python/tree/main/Image%20Analysis/otsu_segmentation/img). 

![Alt text](https://raw.githubusercontent.com/SamHSoftware/Python/main/Image%20Analysis/otsu_segmentation/img/nuclei.png)

## Outputs:

In order to segment the image, and generate the script outputs, run the following code:
```
# Function input arg 1: The file path of the image which needs to be segmented. 
# Function input arg 2: 'True' to display the original image, the histogram and the segmentation output. 'False' to not display the output.
# Function output 1: The value of the threshold calculated by the Otsu algorith.
# Function output 2: A segmented image of type int8, with background = 0, and foreground = 255. 
threshold, segmented_image = otsu_segment(file_path, True)
```

There are three outputs which this script will generate.  

### Output 1: 
The value of the threshold calculated by the Otsu algorith, assigned to ```threshold```.

### Output 2: 
The script will assign the segmented image to ```segmented_image```. This image will be automatically saved.

![Alt text](https://github.com/SamHSoftware/Python/blob/main/Image%20Analysis/otsu_segmentation/img/nuclei_segmented.png?raw=true)

The name of the image will be the original image's name, with the string '\_segmented' appended to the end of the name. The segmented image will share the file extension held by the original image. 

### Output 3: 
Providing ```arg 2``` is set to ```True```, then the code will display a figure with the original image, a histogram with the distribution of pixel values and the segmented image. The segmentation threshold is determined by the Otsu threshold, the value of which is represented by ther vertical red line on the histogram.  

![Alt text](https://github.com/SamHSoftware/Python/blob/main/Image%20Analysis/otsu_segmentation/img/montage.png?raw=true)

## Testing the ```otsu_segment()``` function 

One can simply run the script named ```test_otsu_segmentation_functions``` and call the function ```test_otsu_segment()```. The outputs are given below. 

```
# Test the function 'otsu_segment()' against the provided output. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_otsu_segment(): 
```

What is tested?  
(1) We check for numerical and shape equality between the segmentation output generated when using 'nuclei.png' to  that of'nuclei_segmented.png'.  
(2) We check for type equality between the segmentation output generated when using 'nuclei.png' to 'nuclei_segmented.png'.  
(3) Ensure that the Otsu threshold calculated for 'nuclei.png' is equal to 93.  
