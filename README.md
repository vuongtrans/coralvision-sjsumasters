# coralvision-sjsumasters
Using Color Thresholding and Contouring to Understand Coral Reef Biodiversity

This project utilizes two computer vision techniques - color thresholding and contouring - to extract organisms by color.

To run this code:
*****************
1. This code runs better on Python 3
2. Install the following Python libraries: numpy, OpenCV, matplotlib, os, time 
3. Place images into a folder /images and place in the same directory as this Python file
   /images/
4. Execute the code by issuing: python3 coral-vision-sjsu-masters.py

Results of the code:
********************
The code will generate the following directories:
1. /HSV/ --> Original images converted to HSV color space
2. /color/Contour --> Contouring with no restrictions 
3. /color/ContourAll --> Contouring with a restriction
4. /color/ContourRect -->  Contouring using bounding boxes
5. /color/Mask --> Binary images from color thresholding application
6. /color/Only --> Colorized images of binary images
7. /color/Removed --> Results from removing colors by most dominant to least dominant

In total, 43 directories will be created.

In addition, an elapsed time will be printed at the end of the code execution.
"Elapsed time in seconds: <###>"
  
^^ TODO: Improve the creation of directories to make it cleaner
         Create sub-images for each bounding boxes generated
