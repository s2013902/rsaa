# rsaa
## Point_cm
In this project, Python code based on the attribute values in the excel table is written to calculate all the above indicators at one time. The data in the table 'value'  in sample folder comes from the operation of extracting values by points in ArcGIS software. This code is applicable to the case of two classifications, and the attribute values must be 0 and 1.

## Raster_cm
In this project, confusion matrix will be created according to input .png files.
The input .png files in two different folders will be convert into array. Then the value of each pixcel in the image will be read in to different values. The number of the value = number of image row * number of image columns. Value is depend on the value of image. 
sklearn.model_selection is imported to get test and train data set from the origin values to reduce data volume.
imblearn.under_sampling is imported to select same number of pixcels of different value. 
test1.png and test2.png are sample image should be put in different folder.
make sure your own images area in the same size which means they have the row number and column number.
