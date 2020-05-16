import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# This function reads in image given its name and type OR path
def readImage(directory, name, extension):
    """ Returns the image """
    image_path = directory + name + "." + extension
    image = cv2.imread(image_path)
    return image, image_path

# This function converts the given image to the HSV colorspace
def convertRGBtoHSV(image):
    """ Returns the image after converting to HSV colorspace """
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image_HSV

# This function converts the given image to the XYZ colorspace
def convertRGBtoXYZ(image):
    """ Returns the image after converting to XYZ colorspace """
    image_XYZ = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    return image_XYZ

# This function saves the image for future reference
def saveImageToFile(image, name, extension, colorspace, directory):
    """ Returns the image path of the image """
    final_directory = directory + "/" + colorspace
    final_image_name = name + "_" + colorspace + "." + extension
    final_image_path = final_directory + "/" + final_image_name
    access_rights = 0o755

    # Check if directory exists yet, if not create, otherwise just save
    if (os.path.isdir(final_directory)):
        os.chdir(final_directory)
        cv2.imwrite(final_image_name, image)
    else:
        os.mkdir(final_directory, access_rights)
        os.chdir(final_directory)
        cv2.imwrite(final_image_name, image)

    return final_image_path

# This function can be used to display a single image
def displayImage(image):
    cv2.imshow('image', image)
    cv2.waitKey(100)

# This function breaks down the image path to usable parts
def decipherImageName(img_name):
    """ Returns the image's name and type """
    img_name, img_extension = os.path.splitext(img_name)
    img_extension = img_extension.replace(".","")
    return img_name, img_extension

# This function converts a given color to its RGB representation
def convertColorStringToRGB(color):
    userColor = clr.name_to_rgb(color)
    return userColor

# Use to display the array of generated images for an original image
def displaySingleImage(contoured_img, contoured_all_img, images, original_img):
    """ Plot single image to a figure """
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 5, 1)
    ax0.axis("off")
    ax0.set_title("Original")
    ax0.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1 = fig.add_subplot(1, 5, 3)
    ax1.axis("off")
    ax1.set_title(images[0][0])
    ax1.imshow(cv2.cvtColor(images[0][1], cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1, 5, 2)
    ax2.axis("off")
    ax2.set_title("w/o Orange")
    ax2.imshow(cv2.cvtColor(images[1][1], cv2.COLOR_BGR2RGB))
    ax3 = fig.add_subplot(1, 5, 4)
    ax3.axis("off")
    ax3.set_title("Contour-ed")
    ax3.imshow(cv2.cvtColor(contoured_img, cv2.COLOR_BGR2RGB))
    ax4 = fig.add_subplot(1, 5, 5)
    ax4.axis("off")
    ax4.set_title("Contour-ed all")
    ax4.imshow(cv2.cvtColor(contoured_all_img, cv2.COLOR_BGR2RGB))
    plt.show()

# This function displays all images in the same plot
def displayImages(images, original_img):
    """ Plot the images to a figure """
    fig = plt.figure()
    ax0 = fig.add_subplot(2,7,1)
    ax0.axis("off")
    ax0.set_title("Original")
    ax0.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1 = fig.add_subplot(2,7,2)
    ax1.axis("off")
    ax1.set_title(images[0][0])
    ax1.imshow(cv2.cvtColor(images[0][1], cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(2,7,3)
    ax2.axis("off")
    ax2.set_title(images[1][0])
    ax2.imshow(cv2.cvtColor(images[1][1], cv2.COLOR_BGR2RGB))
    ax3 = fig.add_subplot(2,7,4)
    ax3.axis("off")
    ax3.set_title(images[2][0])
    ax3.imshow(cv2.cvtColor(images[2][1], cv2.COLOR_BGR2RGB))
    ax4 = fig.add_subplot(2,7,5)
    ax4.axis("off")
    ax4.set_title(images[3][0])
    ax4.imshow(cv2.cvtColor(images[3][1], cv2.COLOR_BGR2RGB))
    ax5 = fig.add_subplot(2,7,6)
    ax5.axis("off")
    ax5.set_title(images[4][0])
    ax5.imshow(cv2.cvtColor(images[4][1], cv2.COLOR_BGR2RGB))
    ax6 = fig.add_subplot(2,7,7)
    ax6.axis("off")
    ax6.set_title(images[5][0])
    ax6.imshow(cv2.cvtColor(images[5][1], cv2.COLOR_BGR2RGB))
    ax7 = fig.add_subplot(2,7,9)
    ax7.axis("off")
    ax7.set_title(images[6][0])
    ax7.imshow(cv2.cvtColor(images[6][1], cv2.COLOR_BGR2RGB))
    ax8 = fig.add_subplot(2,7,10)
    ax8.axis("off")
    ax8.set_title(images[7][0])
    ax8.imshow(cv2.cvtColor(images[7][1], cv2.COLOR_BGR2RGB))
    ax9 = fig.add_subplot(2,7,11)
    ax9.axis("off")
    ax9.set_title(images[8][0])
    ax9.imshow(cv2.cvtColor(images[8][1], cv2.COLOR_BGR2RGB))
    ax10 = fig.add_subplot(2,7,12)
    ax10.axis("off")
    ax10.set_title(images[9][0])
    ax10.imshow(cv2.cvtColor(images[9][1], cv2.COLOR_BGR2RGB))
    ax11 = fig.add_subplot(2,7,13)
    ax11.axis("off")
    ax11.set_title(images[10][0])
    ax11.imshow(cv2.cvtColor(images[10][1], cv2.COLOR_BGR2RGB))
    ax12 = fig.add_subplot(2,7,14)
    ax12.axis("off")
    ax12.set_title(images[11][0])
    ax12.imshow(cv2.cvtColor(images[11][1], cv2.COLOR_BGR2RGB))
    plt.show()

# This function calculates the number of pixels per color (red, orange, yellow, green, purple and pink)
def calculatePixelsByColor(img):
    """Return list of pixel counts for a range of colors"""

    # Image is BGR format
    rows, cols, _ = img.shape
    # Convert image from BGR colorspace to HSV colorspace
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Counter for each color
    color_Red = 0
    color_Orange = 0
    color_Yellow = 0
    color_Green = 0
    color_Purple = 0
    color_Pink = 0
    color_Neutral = 0

    # Uses table of thresholds that can be found in the report
    for i in range(rows):
        for j in range(cols):
            k = imgHSV[i,j]
            if k[0] >= 0 and k[0] <= 8 and k[1] >=  65 and k[1] <= 255 and k[2] >= 95 and k[2] <= 255:
                color_Red = color_Red + 1
            if k[0] > 8 and k[0] <= 18 and k[1] >= 120 and k[1] <= 255 and k[2] >= 50 and k[2] <= 255:
                color_Orange = color_Orange + 1
            if k[0] > 18 and k[0] <= 27 and k[1] >= 50 and k[1] <= 255 and k[2] >= 20 and k[2] <= 255:
                color_Yellow = color_Yellow + 1
            if k[0] > 27 and k[0] <= 100 and k[1] >= 50 and k[1] <= 255 and k[2] >= 10 and k[2] <= 255:
                color_Green = color_Green + 1
            if k[0] > 130 and k[0] <= 150 and k[1] >= 20 and k[1] <= 255 and k[2] >= 20 and k[2] <= 255:
                color_Purple = color_Purple + 1
            if k[0] > 150 and k[0] <= 180 and k[1] >= 50 and k[1] <= 255 and k[2] >= 50 and k[2] <= 255:
                color_Pink = color_Pink + 1
            color_Neutral = color_Neutral + 1
        # End FOR J loop
    # End FOR I loop

    # Calculate the percentage of each color in given image
    pix_total = rows * cols
    red_pixels = color_Red/pix_total
    orange_pixels = color_Orange/pix_total
    yellow_pixels = color_Yellow/pix_total
    green_pixels = color_Green/pix_total
    purple_pixels = color_Purple/pix_total
    pink_pixels = color_Pink/pix_total
    neutral_pixels = color_Neutral/pix_total

    # Stores the percentages to be used to determine the
    # most dominant and the least dominant colors (minus neutral and blue)
    pixel_counts = []
    pixel_counts.append(('Grey', 10.0))
    pixel_counts.append(('Red', red_pixels))
    pixel_counts.append(('Orange', orange_pixels))
    pixel_counts.append(('Yellow', yellow_pixels))
    pixel_counts.append(('Green', green_pixels))
    pixel_counts.append(('Purple', purple_pixels))
    pixel_counts.append(('Pink', pink_pixels))

    sorted_pixel_counts = sorted(pixel_counts, key=lambda x: x[1], reverse=True)

    return sorted_pixel_counts

# TODO: Save bounding box as an image using the generated coordinates
# def saveBoundingBox(img, coordinates)
#     return img

def main():
    """ Calls various functions to convert, threshold, and contour images """

    # Start time of program
    start = time.time()

    # Get current working directory and work with images in desired directory
    cwd = os.getcwd()

    # Data sets of images
    original = cwd + '/images/'

    # Loop through each photo in selected directory
    for file in os.listdir(original):
        if file.endswith(".JPG"):

            # Determine image name and file extension
            img_name, img_extension = decipherImageName(file)

            print ('>>> Begin: Converting image to various colorspaces <<<')

            # Read image and return image & image's full path
            img, img_path = readImage(original, img_name, img_extension)
            original_img = img

            # Convert images to other colorspaces (HSV, XYZ)
            img_HSV = convertRGBtoHSV(img)

            # Save newly created colorspace converted image to respective directory
            directory_to_save = cwd;
            saveImageToFile(img_HSV, img_name, img_extension, "HSV", directory_to_save)

            print ('>>> End: Converting image to various colorspaces <<<')

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - #

            print ('>>> Begin: Color Thresholding - Find Dominant Colors <<<')
            sorted_pixel_counts = calculatePixelsByColor(img)
            print ('>>> End: Color Thresholding - Find Dominant Colors <<<')

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - #

            # Thresholds for each color
            hsv_color_ranges = [
                ('Red', (0, 40, 95), (8, 255, 255)),
                ('Orange', (8, 120, 50), (18, 255, 255)),
                ('Yellow', (18, 50, 50), (27, 255, 255)),
                ('Green', (27, 40, 20), (90, 255, 255)),
                ('Blue', (100, 50, 50), (130, 255, 255)),
                ('Purple', (130, 20, 20), (150, 255, 255)),
                ('Pink', (150, 50, 50), (180, 255, 255)),
                ('Grey', (0, 0, 20), (180, 50, 255))
            ]

            # Get number of colors
            length = len(sorted_pixel_counts)
            images = []
            final_image = ''
            # Apply color thresholding and contouring for each color
            for i in range(length):

                print ('>>> Begin: Color Thresholding <<<')

                current_color = sorted_pixel_counts[i][0]
                res = [i for i in hsv_color_ranges if current_color in i ]
                hsv_lower = res[0][1]
                hsv_higher = res[0][2]

                # Get original image
                if final_image == '':
                    img1 = img
                # Otherwise use current version of original image
                else:
                    img1 = cv2.imread(final_image)
                hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
                result = img1.copy()
                contour_img = img1.copy()
                contour_img_all = img1.copy()
                mask = cv2.inRange(hsv, hsv_lower, hsv_higher)
                saveImageToFile(mask, img_name, img_extension, current_color+"-Mask", directory_to_save)
                res = cv2.bitwise_and(result, result, mask = mask)
                res_inverted = cv2.bitwise_not(result, result, mask = mask)
                images.append((current_color, res))
                images.append((current_color, res_inverted))
                saveImageToFile(res, img_name, img_extension, current_color+"-Only", directory_to_save)
                saveImageToFile(res_inverted, img_name, img_extension, current_color+"-Inverted", directory_to_save)
                rows = mask.shape[0]
                cols = mask.shape[1]

                # Convert every white pixel to a gray pixel
                for x in range(0, rows):
                    for y in range(0, cols):
                        if mask[x][y] != 0:
                            res_inverted[x][y] = [255, 255, 255]

                final_image_path = saveImageToFile(res_inverted, img_name, img_extension, current_color+"-Removed", directory_to_save)
                final_image = final_image_path

                print ('>>> End: Color Thresholding <<<')

                print ('>>> Begin: Segmentation/Contouring Ideas <<<')

                contour_rect = contour_img.copy()

                # Do regular contouring
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 25000:
                        cv2.drawContours(contour_img, contour,
                                         contourIdx = -1,
                                         color = (0, 255, 0),
                                         thickness = 3)
                    cv2.drawContours(contour_img_all, contour,
                                     contourIdx = -1,
                                     color = (0, 255, 0),
                                     thickness = 3)
                saveImageToFile(contour_img, img_name, img_extension, current_color+"-Contour", directory_to_save)
                saveImageToFile(contour_img_all, img_name, img_extension, current_color+"-Contour-All", directory_to_save)

                # Do rectangle contouring
                for contour in contours:
                    if cv2.contourArea(contour) > 25000:
                        # Obtain the bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(contour_rect, (x,y), (x+w, y+h), (0, 255, 0), 6)
                # save new image with rectangle contours
                saveImageToFile(contour_rect, img_name, img_extension, current_color+"-Contour-Rect", directory_to_save)

                print ('>>> End: Segmentation/Contouring Ideas <<<')

            # END of FOR LOOP
        # END IF loop
    # END FOR loop

    end = time.time()
    elapsed = end - start
    print("Elapsed time in seconds: ", elapsed)

###########
if __name__ == "__main__":
    main()
else:
    print("The module coral-vision.py is intended to be executed to apply color thresholding and contouring on the images of the ARMS structure.")
###########
