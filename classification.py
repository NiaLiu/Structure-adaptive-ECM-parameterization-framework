from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import numpy as np
import random

# data
excluded = False
choose = 'd2'

exclusion = ''
if excluded:
    exclusion = '-exclude'

def shuffle_data(x,y):
    length = len(x)
    if len(x) != len(y):
        exit(1)
    else:
        index = np.random.permutation(length)
        return x[index], y[index]
def unnormalize(lst,l1,l2):
        # arr = np.asarray(lst, dtype=float)
        arr=np.array(lst,dtype=np.float64)
        return arr * (l2 - l1) + l1
def x(arr, frequency):
    n_pairs = len(arr) // 2
    out = [v for p in range(0, n_pairs, frequency)
           for v in (arr[2*p], arr[2*p + 1])]
    return np.array(out)


if excluded:
    data = np.load("DataSet/classification-data{}/data.npz".format(exclusion+choose))
    Xtr,ytr = data['x'], data['y']
    Xtr,ytr = shuffle_data(Xtr,ytr)
    ytr = LabelEncoder().fit_transform(ytr)
    data_test = np.load("DataSet/classification-data{}/data_test.npz".format(exclusion+choose))
    Xva,yva = data_test['x'], data_test['y']
    Xva,yva = shuffle_data(Xtr,ytr)
    yva = LabelEncoder().fit_transform(yva)
    # y = LabelEncoder().fit_transform(y)
else:
    data = np.load("DataSet/classification-data/data.npz")
    Xtr,ytr = data['x'], data['y']
    ytr = LabelEncoder().fit_transform(ytr)
    # Xtr, ytr = shuffle_data(Xtr, ytr)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.1, random_state=77, stratify=ytr, shuffle=True)
    print("{} data samples for training, {} data samples for testing".format(len(Xtr), len(Xva)))

# model
clf = XGBClassifier(
    n_estimators=1300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.7,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=416000,)

clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

# print(len(clf.get_params()))
# evaluate
import time
start = time.time()
pred = clf.predict(Xva[:1])
print('Run time: ',time.time()-start)
exit(0)
cm = confusion_matrix(yva, pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()
frequency = 11

n1_min = np.array([ 0.0202  , -0.127383,  0.0203  , -0.119617,  0.020588, -0.115298,
        0.0208  , -0.112031,  0.021181, -0.106872,  0.0215  , -0.097925,
        0.021708, -0.085198,  0.021731, -0.083686,  0.021778, -0.083212,
        0.021871, -0.074036,  0.022034, -0.060257,  0.022268, -0.046013,
        0.022525, -0.033938,  0.022737, -0.02466 ,  0.022874, -0.017897,
        0.022949, -0.0214  ,  0.022986, -0.026355,  0.023005, -0.036617,
        0.023015, -0.050887,  0.023022, -0.070724,  0.023029, -0.09465 ,
        0.023041, -0.119862,  0.02306 , -0.140234,  0.023095, -0.146814,
        0.023156, -0.137622,  0.023265, -0.119498,  0.023469, -0.119706,
        0.023698, -0.175416,  0.023359, -0.259784,  0.023065, -0.383719])
n1_max = np.array([ 2.77673e-01,  3.50200e-03,  2.52743e-01,  2.31100e-03,
        2.26213e-01,  2.67000e-03,  2.14145e-01,  9.85000e-04,
        2.06287e-01,  6.27000e-04,  1.93497e-01,  3.68000e-04,
        1.73389e-01,  1.69000e-04,  1.46487e-01, -3.24000e-06,
        1.16091e-01, -1.64000e-04,  8.88200e-02, -1.66000e-04,
        7.23350e-02, -1.38000e-04,  6.27280e-02, -1.26000e-04,
        5.40520e-02, -1.31000e-04,  4.92000e-02, -1.54000e-04,
        5.29000e-02, -2.02000e-04,  5.73000e-02, -2.65000e-04,
        6.34000e-02, -3.32000e-04,  7.20000e-02, -3.82000e-04,
        8.16000e-02, -4.48000e-04,  9.14000e-02, -5.58000e-04,
        9.95000e-02, -4.16000e-04,  1.21208e-01, -2.90000e-04,
        1.56750e-01, -2.02000e-04,  2.07418e-01, -1.42000e-04,
        2.56680e-01, -1.00000e-04,  2.94884e-01, -7.15000e-05,
        3.21698e-01, -5.14000e-05,  3.41202e-01,  4.96000e-03,
        3.53924e-01,  6.27000e-03,  3.64510e-01,  7.69000e-03])
n1_min,n1_max=x(n1_min,frequency),x(n1_max,frequency) 
X_original = unnormalize(Xva,n1_min,n1_max)

print("TP={}, TN={}, FP={}, FN={}".format(tp,tn,fp,fn))
print("Accuracy:", accuracy_score(yva, pred))
print(classification_report(yva, pred))