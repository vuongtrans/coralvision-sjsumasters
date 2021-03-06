# coralvision-sjsumasters
Using Color Thresholding and Contouring to Understand Coral Reef Biodiversity

This project utilizes two computer vision techniques - color thresholding and contouring - to extract organisms by color.

!!! Before running any of the modules !!!
Please pip3 install: OpenCV, matplotlib, numpy, pandas.
Python should had come preinstalled with Path, os, time libraries

- - - - - - - - - - - - - -
Notes for PlateOccupancy.py
- - - - - - - - - - - - - - 
This module is intended solely to calculate how much of the plate is occupied.
After cloning the repository, place all folders of images into /images/, the code will iterate through all subdirectories of /images/ and apply the calculation to each image. 

The results will be saved to a csv file with the format:
ImageName, PercentageOccupied
<name of file>, 10
<name of file>, 20
   
To run this code:
1. Execute the code by issuing: python3 PlateOccupancy.py

Results of the code:
A results.csv file containing the plates' name including the percentage of the plate that is unoccupied.

- - - - - - - - - - - - - - - - - - - 
Notes for CoralVisionSjsuMasters.py
- - - - - - - - - - - - - - - - - - - 
This module is intended solely to apply color thresholding and contouring to the images of the plate.

To run this code:
1. This code runs better on Python 3
2. Install the following Python libraries: numpy, OpenCV, matplotlib, os, time 
3. Place images into a folder /images and place in the same directory as this Python file
   /images/
4. Execute the code by issuing: python3 CoralVisionSjsuMasters.py

Results of the code:
The code will generate the following directories:
1. /HSV/ --> Original images converted to HSV color space
2. /*color*/Contour --> Contouring with no restrictions 
3. /*color*/ContourAll --> Contouring with a restriction
4. /*color*/ContourRect -->  Contouring using bounding boxes
5. /*color*/Mask --> Binary images from color thresholding application
6. /*color*/Only --> Colorized images of binary images
7. /*color*/Removed --> Results from removing colors by most dominant to least dominant

In total, 43 directories will be created. Folders 2-7 will be created for each of the seven colors

In addition, an elapsed time will be printed at the end of the code execution.
"Elapsed time in seconds: <###>"
  
^^ TODO: Improve the creation of directories to make it cleaner
         Create sub-images for each bounding boxes generated
