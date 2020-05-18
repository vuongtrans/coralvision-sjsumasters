import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from CoralVisionSjsuMasters import decipherImageName, readImage, saveImageToFile, convertRGBtoHSV

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

    return pixel_counts

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

    # Loop through each photo in selected directory
    for file in os.listdir(original):
        if file.endswith(".JPG"):
            # Determine image name and file extension
            img_name, img_extension = decipherImageName(file)

            # Read image and return image & image's full path
            img, img_path = readImage(original, img_name, img_extension)

            img_HSV = convertRGBtoHSV(img)

            # Save newly created colorspace converted image to respective directory
            directory_to_save = cwd;
            saveImageToFile(img_HSV, img_name, img_extension, "HSV", directory_to_save)

            grey_pixels = calculateGreyPixels(img)
            length = len(grey_pixels)
            for i in range(length):
                current_color = grey_pixels[i][0]
                res = [i for i in gray_threshold if current_color in i ]
                hsv_lower = res[0][1]
                hsv_higher = res[0][2]

                img1 = img
                hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                result = img1.copy()
                mask = cv2.inRange(hsv, hsv_lower, hsv_higher)
                saveImageToFile(mask, img_name, img_extension, current_color+"-Mask", directory_to_save)
                res = cv2.bitwise_and(result, result, mask = mask)
                saveImageToFile(res, img_name, img_extension, current_color+"-Only", directory_to_save)

        # END IF loop
    # END FOR loop

###########
if __name__ == "__main__":
    main()
else:
    print("The module plate_approximation.py is intended to be executed to determine the percentage of unoccupied space on a plate.")
###########
