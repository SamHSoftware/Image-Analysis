# Import any necessary packages and modules. 
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np 
import scipy.ndimage

# Test the function 'comparing_filters()' against the provided output. 
# Function input args: None. 
# Function returns: When no errors are detected, a statement confirming this is printed. When errors are detcted, assertion errors are raised. 
def test_compare_edge_detection(): 

    # Create a CLAHE object for use later on. 
    clahe = cv.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    
    # Reconstruct the path so that we can load in the provided images from the 'img' folder. 
    cwd = os.getcwd()
    new_path = cwd.replace('edge-detection-package', 'img')

    ######### Test set 1: Testing the initial gaussian filtering.
    original = cv.imread(f"{new_path}/original.png", 0) 
    gaussian_filtered = cv.imread(f"{new_path}/gaussian_filtered.png", 0) 

    # Test 1.1: Test to see if the original image has loaded in correctly.
    if original is None: 
        raise TypeError("Test 1.1 failed. Error opening image. Image 'original' is of type None") 

    # Test 1.2 Test to see if the gaussian_filtered image has loaded in correctely. 
    if gaussian_filtered is None: 
        raise TypeError("Test 1.2 failed. Error opening image. Image 'gaussian_filtered' is of type None") 
    
    # Test 1.3: Check for numerical and shape equality between the newly gaussian-smoothened image (named 'new_gaussian') and it's unit testing counterpart (saved as 'gaussian_filtered'). 
    new_gaussian = scipy.ndimage.gaussian_filter(original, 3) 
    new_gaussian = clahe.apply(new_gaussian)
    assert np.array_equal(gaussian_filtered, new_gaussian), "Test 1.3 failed. The newly gaussian-smoothened image (named 'new_gaussian') and it's unit testing counterpart (named 'gaussian_filtered') have either unequal dimensions or numerical vales."

    # Test 1.4: Check for type equality between the newly gaussian-smoothened image (named 'new_gaussian') and it's unit testing counterpart (saved as 'gaussian_filtered').
    assert gaussian_filtered.dtype == new_gaussian.dtype, "Test 1.4 failed. The newly gaussian-smoothened image (named 'new_gaussian') and it's unit testing counterpart (saved as 'gaussian_filtered') have different data types."

    ######### Test set 2: Testing the laplacian filtering.
    laplacian_filtered = cv.imread(f"{new_path}/laplacian_filtered.png", 0) 
    
    # Test 2.1: Test to see if the laplacian_filtered image has loaded in correctly.
    if laplacian_filtered is None: 
        raise TypeError("Test 2.1 failed. Error opening image. Image 'laplacian_filtered' is of type None") 
    
    # Test 2.2: Check for numerical and shape equality between the newly laplacian-filtered image (named 'new_laplacian') and it's unit testing counterpart (saved as 'laplacian_filtered'). 
    new_laplacian = cv.Laplacian(gaussian_filtered, cv.CV_8U, 3)
    new_laplacian = clahe.apply(new_laplacian)
    assert np.array_equal(new_laplacian, laplacian_filtered), "Test 2.3 failed. The newly laplacian-filtered image (named 'new_laplacian') and it's unit testing counterpart (named 'laplacian_filtered') have either unequal dimensions or numerical vales."

    # Test 2.3: Check for type equality between the newly laplacian-filtered image (named 'new_laplacian') and it's unit testing counterpart (saved as 'laplacian_filtered'). 
    assert new_laplacian.dtype == laplacian_filtered.dtype, "Test 1.4 failed. The newly laplacian-filtered image (named 'new_laplacian') and it's unit testing counterpart (saved as 'laplacian_filtered') have different data types."

    ######### Test set 3: Testing the Sobel-x filtering.
    sobel_x_filtered = cv.imread(f"{new_path}/sobel_x_filtered.png", 0) 
    
    # Test 3.1: Test to see if the sobel_x_filtered image has loaded in correctly.
    if sobel_x_filtered is None: 
        raise TypeError("Test 3.1 failed. Error opening image. Image 'sobel_x_filtered' is of type None") 
    
    # Test 3.2: Check for numerical and shape equality between the newly sobel-x-filtered image (named 'new_sobel_x') and it's unit testing counterpart (saved as 'sobel_x_filtered'). 
    new_sobel_x = cv.convertScaleAbs(cv.Sobel(gaussian_filtered, cv.CV_16S, 1, 0, ksize=3))
    new_sobel_x2 = clahe.apply(new_sobel_x)
    assert np.array_equal(new_sobel_x2, sobel_x_filtered), "Test 3.3 failed. The newly sobel-x filtered image (named 'new_sobel_x') and it's unit testing counterpart (named 'sobel_x_filtered') have either unequal dimensions or numerical vales."

    # Test 3.3: Check for type equality between the newly sobel-x-filtered image (named 'new_sobel_x') and it's unit testing counterpart (saved as 'sobel_x_filtered'). 
    assert new_sobel_x2.dtype == sobel_x_filtered.dtype, "Test 3.4 failed. The newly sobel-x filtered image (named 'new_sobel_x') and it's unit testing counterpart (named 'sobel_x_filtered') have either unequal dimensions or numerical vales."
   
    ######### Test set 4: Testing the Sobel-y filtering.
    sobel_y_filtered = cv.imread(f"{new_path}/sobel_y_filtered.png", 0) 
    
    # Test 4.1: Test to see if the sobel_y_filtered image has loaded in correctly.
    if sobel_y_filtered is None: 
        raise TypeError("Test 4.1 failed. Error opening image. Image 'sobel_y_filtered' is of type None") 
    
    # Test 4.2: Check for numerical and shape equality between the newly sobel-y-filtered image (named 'new_sobel_y') and it's unit testing counterpart (saved as 'sobel_y_filtered'). 
    new_sobel_y = cv.convertScaleAbs(cv.Sobel(gaussian_filtered, cv.CV_16S, 0, 1, ksize=3))
    new_sobel_y2 = clahe.apply(new_sobel_y)
    assert np.array_equal(new_sobel_y2, sobel_y_filtered), "Test 4.3 failed. The newly sobel-y filtered image (named 'new_sobel_y') and it's unit testing counterpart (named 'sobel_y_filtered') have either unequal dimensions or numerical vales."

    # Test 4.3: Check for type equality between the newly sobel-y-filtered image (named 'new_sobel_y') and it's unit testing counterpart (saved as 'sobel_y_filtered'). 
    assert new_sobel_y2.dtype == sobel_y_filtered.dtype, "Test 4.4 failed. The newly sobel-y filtered image (named 'new_sobel_y') and it's unit testing counterpart (named 'sobel_y_filtered') have either unequal dimensions or numerical vales."
    
    ######### Test set 5: Testing the Sobel-x-y sum.
    sobel_x_y_sum = cv.imread(f"{new_path}/sobel_x_y_sum.png", 0) 
    
    # Test 5.1: Test to see if the sobel_x_y_filtered image has loaded in correctly.
    if sobel_x_y_sum is None: 
        raise TypeError("Test 5.1 failed. Error opening image. Image 'sobel_x_y_sum' is of type None") 
    
    # Test 5.2: Check for numerical and shape equality between the newly sobel-x-y-filtered image (named 'new_sobel_x_y_sum') and it's unit testing counterpart (saved as 'sobel_x_y_sum'). 
    new_sobel_x_y_sum = cv.addWeighted(new_sobel_x, 0.5, new_sobel_y, 0.5, 0)
    new_sobel_x_y_sum = clahe.apply(new_sobel_x_y_sum)
    assert np.array_equal(new_sobel_x_y_sum, sobel_x_y_sum), "Test 5.3 failed. The newly filtered image (named 'new_sobel_x_y_sum') and it's unit testing counterpart (named 'sobel_x_y_sum') have either unequal dimensions or numerical vales."

    # Test 5.3: Check for type equality between the newly sobel-x-y-filtered image (named 'new_sobel_x_y_sum') and it's unit testing counterpart (saved as 'sobel_x_y_sum'). 
    assert new_sobel_x_y_sum.dtype == sobel_x_y_sum.dtype, "Test 5.4 failed. The newly filtered image (named 'new_sobel_x_y_sum') and it's unit testing counterpart (named 'sobel_x_y_sum') have either unequal dimensions or numerical vales."
    
    ######### Test set 6: Testing the canny filtering.
    canny_filtered = cv.imread(f"{new_path}/canny_filtered.png", 0) 
    
    # Test 5.1: Test to see if the sobel_x_y_filtered image has loaded in correctly.
    if canny_filtered is None: 
        raise TypeError("Test 6.1 failed. Error opening image. Image 'canny_filtered' is of type None") 
    
    # Test 6.2: Check for numerical and shape equality between the newly canny-filtered image (named 'new_canny_filtered') and it's unit testing counterpart (saved as 'canny_filtered'). 
    new_canny_filtered = cv.Canny(gaussian_filtered, 40, 45)
    assert np.array_equal(canny_filtered, new_canny_filtered), "Test 6.3 failed. The newly canny-filtered image (named 'new_canny_filtered') and it's unit testing counterpart (named 'canny_filtered') have either unequal dimensions or numerical vales."

    # Test 5.3: Check for type equality between the newly canny-filtered image (named 'new_canny_filtered') and it's unit testing counterpart (saved as 'canny_filtered'). 
    assert canny_filtered.dtype == new_canny_filtered.dtype, "Test 6.4 failed. The newly canny-filtered image (named 'new_canny_filtered') and it's unit testing counterpart (named 'canny_filtered') have either unequal dimensions or numerical vales."
    
    print('Tests complete. No errors found.')
    
# Run the function for unit testing
test_compare_edge_detection()