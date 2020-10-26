from tkinter import *
from tkinter import filedialog
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.ndimage
import ntpath
import os

# A function to allow the user to select the image they wish to analyse. 
# Function input args: none. 
# Function output 1: The file path of the image in question. 
def file_selection_dialog():
    root = Tk()
    root.title('Please select the file in question')
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=[("All files", "*.*")])
    file_path = root.filename
    root.destroy()
    
    # If the image has more channels than one, throw an error. 
    image = cv.imread(file_path, 0) # We load in the image. 
    if (len(image.shape) > 2): 
        raise TypeError("Your image has more than one channel. Please select an image with only one channel.")
    return file_path

# A function to compare the effects of different image filters.
# Function input arg 1: file_path --> A string containing the file path to the image in question. 
# Function input arg 2: plot_images --> Set to True or False. If True, a montage of filtered images will be displayed in the console. 
# Function input arg 3: save_plot --> Set to True or False. If True, a montage of filtered images will be saved to the same directory as the source image.
# Function input arg 4: kernel_size --> Set to positive integer representing size of gaussian kernel when smoothing image. 
# Function output 1: The montage of filtered images. 
def compare_edge_detection(file_path, plot_images, save_plot, kernel_size):

    # Load in the image. 
    image = cv.imread(file_path, 0) 
    
    # Check to see that if the image loaded in without issue. 
    if image is None: 
        raise TypeError("Error opening image. Image is of type None") 
    
    # If the image isn't of 'uint-8', then convert it to such.
    if image.dtype != 'uint8': 
        image.astype('uint8')
        print("Converting image to type 'uint8'.")
    
    # Create a CLAHE object for use later on. 
    clahe = cv.createCLAHE(clipLimit=1, tileGridSize=(8,8))

    # Gaussian filter image to remove salt and pepper noise, as this can interfere with edge detection.  
    gaussian_filter_image = scipy.ndimage.gaussian_filter(image, kernel_size) 
    gaussian_filter_image = clahe.apply(gaussian_filter_image)
    
    # Filter 1: Laplacian edge detection. 
    laplacian_edges = cv.Laplacian(gaussian_filter_image, cv.CV_8U, kernel_size)
    laplacian_edges = clahe.apply(laplacian_edges)
    
    # Filter 2: Sobel-x edge detection 
    sobel_x = cv.Sobel(gaussian_filter_image, cv.CV_16S, 1, 0, ksize=3)
    sobel_x_abs = cv.convertScaleAbs(sobel_x)  
    sobel_x_abs2 = clahe.apply(sobel_x_abs)
    
    # Filter 3: Sobel-y edge detection 
    sobel_y = cv.Sobel(gaussian_filter_image, cv.CV_16S, 0, 1, ksize=3)
    sobel_y_abs = cv.convertScaleAbs(sobel_y)
    sobel_y_abs2 = clahe.apply(sobel_y_abs)
    
    # (Pseudo)Filter 4 : Equally weighted sum of Sobel-x and -y. 
    sobel_x_y = cv.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)
    sobel_x_y = clahe.apply(sobel_x_y)
    
    # Filter 6: Canny Edge detection.
    canny_edges = cv.Canny(gaussian_filter_image, 40, 45)
    
    # Plot the filtered images alongside their segmented counterparts. 
    fig, axs = plt.subplots(3,2,figsize=(8,15))

    axs[0,0].imshow(gaussian_filter_image, cmap='gray') # Plot the original unmodified image.
    axs[0,0].set_title('Gaussian-blurred image')
    axs[0,0].axis('off')
                             
    axs[0,1].imshow(laplacian_edges, cmap='gray') # Plot the laplacian edge detection result. 
    axs[0,1].set_title('Laplacian edge detection')
    axs[0,1].axis('off')

    axs[1,0].imshow(sobel_x_abs2, cmap='gray') # Plot the Sobel-x edge detection result. 
    axs[1,0].set_title('Sobel-x edge detection')
    axs[1,0].axis('off')
    
    axs[1,1].imshow(sobel_y_abs2, cmap='gray') # Plot the Sobel-y edge detection result.
    axs[1,1].set_title('Sobel-y edge detection')
    axs[1,1].axis('off')

    axs[2,0].imshow(sobel_x_y, cmap='gray') # Plot the equally weighted sum of Sobel-x and -y edge detections. 
    axs[2,0].set_title('The equally weighted sum\n of Sobel-x and -y edge detections')
    axs[2,0].axis('off')
    
    axs[2,1].imshow(canny_edges, cmap='gray') # Plot the Canny edge detection result.
    axs[2,1].set_title('Canny edge detection')
    axs[2,1].axis('off')

    plt.subplots_adjust(wspace=0.1)
    fig = plt.gcf()
    
    # Save the plot if the user desires it.
    if save_plot:
        _, tail = ntpath.split(file_path)
        filename, _ = os.path.splitext(tail)
        new_file_path = file_path.replace(tail, f"{filename}_comparing_edge_detection_filters.png")
        plt.savefig(new_file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (plot_images == False):
        plt.close()
    
    return(fig)

    