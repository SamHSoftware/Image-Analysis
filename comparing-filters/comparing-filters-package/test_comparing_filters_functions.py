# Import the module containing the functions we need to unit test. 
import comparing_filters_functions

# Import any necessary packages and modules. 
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np 
import scipy.ndimage

# Test the function 'comparing_filters()' against the provided output. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_compare_filters(): 

    # Reconstruct the path so that we can load in the provided images from the 'img' folder. 
    cwd = os.getcwd()
    new_path = cwd.replace('comparing-filters-package', 'img')

    ######### Test set 1: Testing the process without filtering.
    original = cv.imread(f"{new_path}/original.png", 0) 
    original_segmented = cv.imread(f"{new_path}/original_segmented.png", 0) 
    threshold, test_seg = cv.threshold(original, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)     

    # Test 1.1: Check for numerical and shape equality between the segmentation output and the unit-test image.
    assert np.array_equal(original_segmented, test_seg), "Test 1.1 failed. The original segmentation output and the test segmentation output have either unequal dimensions or numerical vales."

    # Test 1.2: Check for type equality between the segmentation output generated and the test unit-image.
    assert original_segmented.dtype == test_seg.dtype, "Test 1.2 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 1.3: Check the Otsu threshold value. 
    assert threshold == 98.0, f"Test 1.3 failed. The newly calculated threshold should have been equal to 98.0 but instead, it was equal to {threshold}"

    ######### Test set 2: Testing the median filtering process.
    median_filtered = cv.imread(f"{new_path}/median_filtered.png", 0) 
    median_filtered_segmented = cv.imread(f"{new_path}/median_filtered_segmented.png", 0) 
    test_filt = scipy.ndimage.median_filter(original, 3) # We apply the filter to 'smooth' the image.
    threshold, test_seg = cv.threshold(test_filt, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)     

    # Test 2.1: Check for numerical and shape equality between the median filtered unit-test image and the current output of the median filter.  
    assert np.array_equal(median_filtered, test_filt), "Test 2.1 failed. The median filtered unit-test image and the current output of the median filter have either unequal dimensions or numerical vales."

    # Test 2.2: Check for type equality between the median filtered unit-test image and the current output of the median filter.  
    assert median_filtered.dtype == test_filt.dtype, "Test 2.2 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 2.3: Check for numerical and shape equality between the segmentation output and the unit-test image.
    assert np.array_equal(median_filtered_segmented, test_seg), "Test 2.3 failed. The original segmentation output and the test segmentation output have either unequal dimensions or numerical vales."

    # Test 2.4: Check for type equality between the segmentation output generated and the unit-test image.
    assert median_filtered_segmented.dtype == test_seg.dtype, "Test 2.4 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 2.5: Check the Otsu threshold value. 
    assert threshold == 93.0, f"Test 2.5 failed. The newly calculated threshold should have been equal to 93.0 but instead, it was equal to {threshold}"

    ######### Test set 3: Testing the gaussian filtering process.
    gaussian_filtered = cv.imread(f"{new_path}/gaussian_filtered.png", 0) 
    gaussian_filtered_segmented = cv.imread(f"{new_path}/gaussian_filtered_segmented.png", 0) 
    test_filt = scipy.ndimage.gaussian_filter(original, 3) # We apply the filter to 'smooth' the image.
    threshold, test_seg = cv.threshold(test_filt, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)     

    # Test 3.1: Check for numerical and shape equality between the median filtered unit-test image and the current output of the median filter.  
    assert np.array_equal(gaussian_filtered, test_filt), "Test 3.1 failed. The median filtered unit-test image and the current output of the median filter have either unequal dimensions or numerical vales."

    # Test 3.2: Check for type equality between the median filtered unit-test image and the current output of the median filter.  
    assert gaussian_filtered.dtype == test_filt.dtype, "Test 3.2 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 3.3: Check for numerical and shape equality between the segmentation output and the unit-test image.
    assert np.array_equal(gaussian_filtered_segmented, test_seg), "Test 3.3 failed. The original segmentation output and the test segmentation output have either unequal dimensions or numerical vales."

    # Test 3.4: Check for type equality between the segmentation output generated and the unit-test image.
    assert gaussian_filtered_segmented.dtype == test_seg.dtype, "Test 3.4 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 3.5: Check the Otsu threshold value. 
    assert threshold == 77.0, f"Test 3.5 failed. The newly calculated threshold should have been equal to 77.0 but instead, it was equal to {threshold}"

    ######### Test set 4: Testing the maximum filtering process.
    maximum_filtered = cv.imread(f"{new_path}/maximum_filtered.png", 0) 
    maximum_filtered_segmented = cv.imread(f"{new_path}/maximum_filtered_segmented.png", 0) 
    test_filt = scipy.ndimage.maximum_filter(original, 3) # We apply the filter to 'smooth' the image.
    threshold, test_seg = cv.threshold(test_filt, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)     

    # Test 4.1: Check for numerical and shape equality between the median filtered unit-test image and the current output of the median filter.  
    assert np.array_equal(maximum_filtered, test_filt), "Test 4.1 failed. The median filtered unit-test image and the current output of the median filter have either unequal dimensions or numerical vales."

    # Test 4.2: Check for type equality between the median filtered unit-test image and the current output of the median filter.  
    assert maximum_filtered.dtype == test_filt.dtype, "Test 4.2 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 4.3: Check for numerical and shape equality between the segmentation output and the unit-test image.
    assert np.array_equal(maximum_filtered_segmented, test_seg), "Test 4.3 failed. The original segmentation output and the test segmentation output have either unequal dimensions or numerical vales."

    # Test 4.4: Check for type equality between the segmentation output generated and the unit-test image.
    assert maximum_filtered_segmented.dtype == test_seg.dtype, "Test 4.4 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 4.5: Check the Otsu threshold value. 
    assert threshold == 117.0, f"Test 4.5 failed. The newly calculated threshold should have been equal to 117.0 but instead, it was equal to {threshold}"

    ######### Test set 5: Testing the minimum filtering process.
    minimum_filtered = cv.imread(f"{new_path}/minimum_filtered.png", 0) 
    minimum_filtered_segmented = cv.imread(f"{new_path}/minimum_filtered_segmented.png", 0) 
    test_filt = scipy.ndimage.minimum_filter(original, 3) # We apply the filter to 'smooth' the image.
    threshold, test_seg = cv.threshold(test_filt, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)     

    # Test 4.1: Check for numerical and shape equality between the median filtered unit-test image and the current output of the median filter.  
    assert np.array_equal(minimum_filtered, test_filt), "Test 5.1 failed. The median filtered unit-test image and the current output of the median filter have either unequal dimensions or numerical vales."

    # Test 4.2: Check for type equality between the median filtered unit-test image and the current output of the median filter.  
    assert minimum_filtered.dtype == test_filt.dtype, "Test 5.2 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 4.3: Check for numerical and shape equality between the segmentation output and the unit-test image.
    assert np.array_equal(minimum_filtered_segmented, test_seg), "Test 5.3 failed. The original segmentation output and the test segmentation output have either unequal dimensions or numerical vales."

    # Test 4.4: Check for type equality between the segmentation output generated and the unit-test image.
    assert minimum_filtered_segmented.dtype == test_seg.dtype, "Test 5.4 failed. The original segmentation output and the test segmentation output have different data types."

    # Test 4.5: Check the Otsu threshold value. 
    assert threshold == 73.0, f"Test 5.5 failed. The newly calculated threshold should have been equal to 73.0 but instead, it was equal to {threshold}"
    
    print('Tests complete, no errors.')