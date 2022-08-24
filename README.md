# Code for accuarcy assessment

This contains sample data and code for accuracy assessment.


## Point_cm
According to the truth and predicted sampleGenerate true positive, false positive, true negative and false negative for accuracy assessment indices calculation. 
This code is applicable to the case of two classifications, and the attribute values must be 0 and 1.

## Raster_cm
Contains function of generate confusion matrix according to input png files.
The input .png files in two different folders will be convert into array. Then the value of each pixcel in the image will be read in to different values. The number of the value = number of image row * number of image columns. 

sklearn.model_selection is imported to get test and train data set from the origin values to reduce data volume.
imblearn.under_sampling is imported to select same number of pixcels of different value. 

The assessed data should be read as:

    LabelPath = r"Data\truth"
    PredictPath = r"Data\merge"

The sample size can be adjusted by:

    a1, b1, c1, d1 = train_test_split(truth, predictt, test_size=0.1, train_size = 0.9, random_state= 3)

test1.png and test2.png are sample image should be put in different folder.
make sure your own images area in the same size which means they have the row number and column number.
