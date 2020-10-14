from tkinter import *
from tkinter import filedialog
import cv2 as cv
import matplotlib.pyplot as plt
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

# Create a function to segment nuclei.
def otsu_segment(file_path, plot_segmentation):
    
    # Load in the image. 
    image = cv.imread(file_path, 0) 
    equalised_image = cv.equalizeHist(image)
    
    # If the image isn't of 'uint-8', then convert it to such. The Otsu function requires the data-type. 
    if image.dtype != 'uint8':
        image.astype('uint8')
        print("Converting image to type 'uint8'.")
    
    # Remove noise from the image. 
    filtered_image = scipy.ndimage.median_filter(image, 3) # We apply the filter to 'smooth' the image.
    
    # Otsu threshold and segment the image. 
    otsu_threshold, segmented_image = cv.threshold(filtered_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Display the segmented image should the user desire it. 
    if plot_segmentation: 
        fig, axs = plt.subplots(ncols = 3, figsize=(8,2.5))

        axs[0].imshow(image, cmap='gray')
        axs[0].set_title('Original image')
        axs[0].axis('off')

        axs[1].hist(image.ravel(), bins=256, range=(0, 255))
        axs[1].set_title('Histogram')
        axs[1].axvline(otsu_threshold, color='r')
        axs[1].set_xlabel('Pixel value \n\nVertical red line represents the Otsu threshold.')
        axs[1].set_ylabel('Count')

        axs[2].imshow(segmented_image, cmap='gray')
        axs[2].set_title('Segmented image')
        axs[2].axis('off')

        plt.subplots_adjust(wspace=0.5)
        #If you want to save the figure, uncomment the line below. 
        #plt.savefig('montage.png', bbox_inches='tight')
        plt.show()
        
    # Save the image with an updated name. 
    head, tail = ntpath.split(file_path)
    filename, file_extension = os.path.splitext(tail)
    newname = f"{filename}_segmented"
    new_file_path = file_path.replace(filename, f"{filename}_segmented")
    cv.imwrite(new_file_path, segmented_image)
    
    return (otsu_threshold, segmented_image)