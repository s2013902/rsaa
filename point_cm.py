import numpy as np
import pandas as pd

io = r"D:\0624\assessment\value.xlsx"
data = pd.read_excel(io, sheet_name = 0, header = 0)
# sheet_name=0 It means to read the first sheet in the excle. The header is the defined column name which is row 0
data1 = np.array(data)

y_true = data1[:, 0] #Measured data column number
y_pred = data1[:, 1] #Sort data column number

# true positive
TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))

# false positive
FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))

# true negative
TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

# false negative
FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))

Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
Accuracy = (TP + TN)/(TP + FP + TN + FN)
Error_rate = (FN + FP)/(TP + FP + TN +FN)
F1_score = 2*Precision*Recall/(Precision + Recall)

po = (TP + TN)/(TP + FP + TN + FN)
pe = (60*(TP+FP) + 60*(FN+TN))/(120*120) #60 is the number of single class samples, 120 is the total number of samples
Kappa = (po - pe)/(1-pe)
Confus_matrix = np.array([[FN, FP], [TN, TP]])

print("Precision:", Precision)
print("Recall:", Recall)
print("Accuracy:", Accuracy)
print("F1_score:", F1_score)
print("Kappa:", Kappa)