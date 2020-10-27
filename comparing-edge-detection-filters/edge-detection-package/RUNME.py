# Import the necessary packages.
from comparing_edge_detection_filters import *

# A function to allow the user to select the image they wish to analyse. 
# Function input args: none. 
# Function output 1: The file path of the image in question. 
file_path = file_selection_dialog()

# A function to compare the effects of different image filters.
# Function input arg 1: file_path --> A string containing the file path to the image in question. 
# Function input arg 2: plot_images --> Set to True or False. If True, a montage of filtered images will be displayed in the console. 
# Function input arg 3: save_plot --> Set to True or False. If True, a montage of filtered images will be saved to the same directory as the source image.
# Function input arg 4: kernel_size --> Set to positive integer representing size of gaussian kernel when smoothing image. 
# Function output 1: The montage of filtered images. 
montage = compare_edge_detection(file_path, plot_images, save_plot, kernel_size)