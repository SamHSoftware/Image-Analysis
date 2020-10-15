# Import functions from the module file. 
from comparing_filters_functions import *

# Function input args: None. 
# Function returns: The file path corresponding to the image selected by the user. 
file_path = file_selection_dialog()

# Function input arg 1: The file path of the image which needs to be segmented. 
# Function input arg 2: 'True' to display the figure which compares the filtered/original images to their segmentation outputs.
# Function input arg 3: 'True' to save the figure which compares the filtered/original images to their segmentation outputs.
# Function input arg 4: The desired colormap to display the images. Must be a string. 
# Function output 1: The figure. 
figure = compare_filters(file_path, plot_images, save_plot, colormap)