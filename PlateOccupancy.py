import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# pip3 install opencv-python
# pip3 install matplotlib

# This function is intended to determine how much of the plate is unoccupied
def calculateGreyPixels(img):
    # Image is BGR format
    rows, cols, _ = img.shape
    # Convert image from BGR colorspace to HSV colorspace
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Counter for each color
    color_Grey = 0
    color_GreyNonPlate = 0
    color_All = 0

    # Uses table of thresholds that can be found in the report
    for i in range(rows):
        for j in range(cols):
            k = imgHSV[i,j]
            if k[0] >= 0 and k[0] <= 180 and k[1] >= 0 and k[1] <= 50 and k[2] >= 20 and k[2] <= 150:
                color_Grey = color_Grey + 1
            if k[0] >= 0 and k[0] <= 180 and k[1] >= 0 and k[1] <= 50 and k[2] >= 151 and k[2] <= 255:
                color_GreyNonPlate = color_GreyNonPlate + 1
            color_All = color_All + 1
        # End FOR J loop
    # End FOR I loop

    # Calculate the percentage of grey pixels on plate (minus the table)
    pix_total = rows * cols
    plate_pixels_total = pix_total - color_GreyNonPlate
    grey_pixels = color_Grey/plate_pixels_total
    print ("The amount of pixels in image: " + str(pix_total))
    print ("The amount of pixels associated with plate: " + str(plate_pixels_total))
    print ("The amount of plate that is unoccupied is " + str(grey_pixels))

    pixel_counts = []
    pixel_counts.append(('Plate', 10.0))
    pixel_counts.append(('Table', 5.0))

    return grey_pixels

# This function breaks down the image path to usable parts
def decipherImageName(filepath):
    """ Returns the image's name and type """
    filepath = os.path.basename(filepath)
    img_name, img_extension = os.path.splitext(filepath)
    img_extension = img_extension.replace(".","")
    return img_name, img_extension

# This function reads in image given its name and type OR path
def readImage(directory, name, extension):
    """ Returns the image """
    image_path = directory + "/" + name + "." + extension
    image = cv2.imread(image_path)
    return image, image_path

def main():
    # Thresholds for grey pixels
    gray_threshold = [
        ('Plate', (0, 0, 20), (180, 50, 150)),
        ('Table', (0, 0, 151), (180, 50, 255))
    ]

    # Get current working directory and work with images in desired directory
    cwd = os.getcwd()

    # Data sets of images
    original = cwd + '/images/'
    
    column_names = ["ImageName", "PercentageUnoccupied"]
    df = pd.DataFrame(columns = column_names)

    for subdir, dirs, files in os.walk(original):
        for file in files:
            filepath = os.path.join(subdir, file)

            # Determine image name and file extension
            img_name, img_extension = decipherImageName(filepath)

            # Read image and return image & image's full path
            img, img_path = readImage(subdir, img_name, img_extension)

            grey_pixels = calculateGreyPixels(img)
            df = df.append({'ImageName': img_name, 'PercentageUnoccupied': grey_pixels}, ignore_index = True)
        # End FOR
    # End FOR

    # Write data to csv file
    df.to_csv('results', index=False)

###########
if __name__ == "__main__":
    main()
else:
    print("The module plate_approximation.py is intended to be executed to determine the percentage of unoccupied space on a plate.")
###########
