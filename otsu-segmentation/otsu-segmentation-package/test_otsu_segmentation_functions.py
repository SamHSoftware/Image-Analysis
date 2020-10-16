# Import the module containing the functions we need to unit test. 
from otsu_segmentation_functions import otsu_segment

# Import any necessary packages and modules. 
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np 

# Test the function 'otsu_segment()' against the provided output. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_otsu_segment(): 
    
    # Load in the images in question. 
    cwd = os.getcwd()
    new_path = cwd.replace('otsu-segmentation-package', 'img')
    nuclei_segmented = cv.imread(f"{new_path}/nuclei_segmented.png", 0) 
    nuclei_path = cwd.replace('otsu-segmentation-package', 'img/nuclei.png')
    threshold, test_segmentation = otsu_segment(nuclei_path, False)
    
    # Test 1: Check for numerical and shape equality between the segmentation output generated when using 'nuclei.png' to 'nuclei_segmented.png'.
    assert np.array_equal(nuclei_segmented, test_segmentation), "The arrays should have the same dimensions and elements. In this case, these conditions are not met"
    
    # Test 2: Check for type equality between the segmentation output generated when using 'nuclei.png' to 'nuclei_segmented.png'.
    assert nuclei_segmented.dtype == test_segmentation.dtype, "The segmentation output (generated when using 'nuclei.png') should be of the same data-type (uint8) to the provided output, 'nuclei_segmented.png'. In this case, they are unfortunately of different data types."
    
    # Test 3: Ensure that the Otsu threshold calculated for 'nuclei.png' is equal to 93. 
    assert threshold == 93, f"The newly calculated threshold should have been equal to 93.0 but instead , it was equal to {threshold}"
    
    print('Tests complete, no errors.')