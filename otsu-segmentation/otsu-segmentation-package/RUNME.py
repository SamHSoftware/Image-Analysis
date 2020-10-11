# Import functions from the module file. 
from otsu_segmentation_functions import *

# Function input args: None. 
# Function returns: The file path corresponding to the image selected by the user. 
file_path = file_selection_dialog()

# Function input arg 1: The file path of the image which needs to be segmented. 
# Function input arg 2: 'True' to display the original image, the histogram and the segmentation output. 'False' to not display the output.
# Function output 1: The value of the threshold calculated by the Otsu algorith.
# Function output 2: A segmented image of type int8, with background = 0, and foreground = 255. 
threshold, segmented_image = otsu_segment(file_path, True)