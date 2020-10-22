from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import cv2 as cv 
import scipy.ndimage
import ntpath
import os

# A function to allow the user to select the image they wish to analyse. 
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
def compare_filters(file_path, plot_images, save_plot, colormap, kernel_size):

    # Load in the image. 
    image = cv.imread(file_path, 0) 
    equalised_image = cv.equalizeHist(image)

    # If the image isn't of 'uint-8', then convert it to such. The Otsu function requires the data-type. 
    if image.dtype != 'uint8':
        image.astype('uint8')
        print("Converting image to type 'uint8'.")

    # Segment the original image. 
    _, image_segmented = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Filter 1: Median filter image. 
    median_filter_image = scipy.ndimage.median_filter(image, kernel_size) # We apply the filter to 'smooth' the image.
    _, median_filter_segmented = cv.threshold(median_filter_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)     
    
    # Filter 2: Gaussian filter image. 
    gaussian_filter_image = scipy.ndimage.gaussian_filter(image, kernel_size) # We apply the filter to 'smooth' the image.
    _, gaussian_filter_segmented = cv.threshold(gaussian_filter_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Filter 3: Maximum filter image. 
    maximum_filter_image = scipy.ndimage.maximum_filter(image, kernel_size) # We apply the filter to 'smooth' the image.
    _, maximum_filter_segmented = cv.threshold(maximum_filter_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Filter 4: Minimum filter image. 
    minimum_filter_image = scipy.ndimage.minimum_filter(image, kernel_size) # We apply the filter to 'smooth' the image.
    _, minimum_filter_segmented = cv.threshold(minimum_filter_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Plot the filtered images alongside their segmented counterparts. 
    fig, axs = plt.subplots(5,2,figsize=(8,20))

    # Plot the original image and it's segmentation output. 
    axs[0,0].imshow(image, cmap=colormap)
    axs[0,0].set_title('Original image')
    axs[0,0].axis('off')
    axs[0,1].imshow(image_segmented, cmap=colormap)
    axs[0,1].set_title('Original image: Segmented')
    axs[0,1].axis('off')

    # Plot the median filtered image and it's segmentation output. 
    axs[1,0].imshow(median_filter_image, cmap=colormap)
    axs[1,0].set_title('Median filtered')
    axs[1,0].axis('off')
    axs[1,1].imshow(median_filter_segmented, cmap=colormap)
    axs[1,1].set_title('Median filtered: Segmented')
    axs[1,1].axis('off')

    # Plot the gaussian filtered image and it's segmentation output. 
    axs[2,0].imshow(gaussian_filter_image, cmap=colormap)
    axs[2,0].set_title('Gaussian filtered')
    axs[2,0].axis('off')
    axs[2,1].imshow(gaussian_filter_segmented, cmap=colormap)
    axs[2,1].set_title('Gaussian filtered: Segmented')
    axs[2,1].axis('off')

    # Plot the maximum filtered image and it's segmentation output. 
    axs[3,0].imshow(maximum_filter_image, cmap=colormap)
    axs[3,0].set_title('Maximum filtered')
    axs[3,0].axis('off')
    axs[3,1].imshow(maximum_filter_segmented, cmap=colormap)
    axs[3,1].set_title('Maximum filtered: Segmented')
    axs[3,1].axis('off')

    # Plot the minimum filterd image and it's segmentation output. 
    axs[4,0].imshow(minimum_filter_image, cmap=colormap)
    axs[4,0].set_title('Minimum filtered')
    axs[4,0].axis('off')
    axs[4,1].imshow(minimum_filter_segmented, cmap=colormap)
    axs[4,1].set_title('Minimum filtered: Segmented')
    axs[4,1].axis('off')

    plt.subplots_adjust(wspace=0)
    fig = plt.gcf()
    
    # Save the plot if the user desires it.
    if save_plot:
        _, tail = ntpath.split(file_path)
        filename, file_extension = os.path.splitext(tail)
        new_file_path = file_path.replace(tail, f"{filename}_comparing_filters.png")
        plt.savefig(new_file_path, dpi=200, bbox_inches='tight')
    
    # Display the plot if the user desires it. 
    if (plot_images == False):
        plt.close()
    
    return(fig)