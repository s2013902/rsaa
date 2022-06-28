import numpy as np
import cv2
import os

""" 
Confusion matrix
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""



def color_dict(labelFolder, classNum):
    colorDict = []
    #  Get the file name in the folder
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  If it is grayscale, convert to RGB
        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  To extract unique values, convert RGB to a number
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        #  Add the unique value of the i pixel matrix to ColorDict
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  Take a unique value for the unique value in the current I pixel matrix
        colorDict = sorted(set(colorDict))
        #  If the number of unique values is equal to the total number of classes (including the background) classnum, stop traversing the remaining images
        if (len(colorDict) == classNum):
            break

    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  Fill the left with zero for the result that does not reach nine digits(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  Blue, Green, Red
        color_BGR = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_BGR.append(color_BGR)
    #  Convert to numpy format
    colorDict_BGR = np.array(colorDict_BGR)
    #  Gray dictionary for storing colors, used for coding during preprocessing
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1, colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY


def ConfusionMatrix(numClass, imgPredict, Label):
    #  Return confusion matrix
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def OverallAccuracy(confusionMatrix):
    #  Return the overall precision (OA)
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA


def Precision(confusionMatrix):
    #   Return precision
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    return precision


def Recall(confusionMatrix):
    #  Return recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    return recall


def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score


def IntersectionOverUnion(confusionMatrix):
    # Return intersection-over-union (IoU)
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    return IoU


def MeanIntersectionOverUnion(confusionMatrix):
    # Return Intersection-over-Union (mIoU)
    intersection = np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return mIoU


def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    # Return Frequency Weighted Intersection-over-Union (FWIoU)
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


#################################################################
#  Label image folder
LabelPath = r"Data\label2"
#  Forecast image folder
PredictPath = r"Data\label1"
#  类别数目(包括背景)
classNum = 2
#################################################################

#  Get category color dictionary
colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

#  Get all images in the folder
labelList = os.listdir(LabelPath)
PredictList = os.listdir(PredictPath)

#  Read first image
Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)

#  Number of images
label_num = len(labelList)

#  Put all the images in an array
label_all = np.zeros((label_num,) + Label0.shape, np.uint8)
predict_all = np.zeros((label_num,) + Label0.shape, np.uint8)
for i in range(label_num):
    Label = cv2.imread(LabelPath + "//" + labelList[i])
    Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
    label_all[i] = Label
    Predict = cv2.imread(PredictPath + "//" + PredictList[i])
    Predict = cv2.cvtColor(Predict, cv2.COLOR_BGR2GRAY)
    predict_all[i] = Predict

#  map the color
for i in range(colorDict_GRAY.shape[0]):
    label_all[label_all == colorDict_GRAY[i][0]] = i
    predict_all[predict_all == colorDict_GRAY[i][0]] = i

#  transform to one dimension
label_all = label_all.flatten()
predict_all = predict_all.flatten()

#  Calculate confusion matrix and precision parameters
confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
precision = Precision(confusionMatrix)
recall = Recall(confusionMatrix)
OA = OverallAccuracy(confusionMatrix)
IoU = IntersectionOverUnion(confusionMatrix)
FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
mIOU = MeanIntersectionOverUnion(confusionMatrix)
f1ccore = F1Score(confusionMatrix)


print("")
print("confusion matrix:")
print(confusionMatrix)
print("precision:")
print(precision)
print("recall:")
print(recall)
print("F1-Score:")
print(f1ccore)
print("OA:")
print(OA)
print("IoU:")
print(IoU)
print("mIoU:")
print(mIOU)
print("FWIoU:")
print(FWIOU)