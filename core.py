import csv
import random
import math
#from utils import mnist_reader
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pickle
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import gzip
import numpy as np

from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DELTA = 127

def pickleLoad(filename):
    with open(filename, "rb") as f:
        filetype = pickle.load(f)
    return filetype

def pickleUnload(filename,filetype):
    with open(filename, "wb") as f:
        pickle.dump(filetype, f)

def classSeparator(r1,r2,X_train,y_train):
    X_train_re = []
    y_train_re = []
    for i in range(0,len(X_train)):
        if (y_train[i] == r1 or y_train[i] == r2):
            y_train_re.append(y_train[i])
            X_train_re.append(X_train[i])
    return (X_train_re, y_train_re)

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def dictmaker(dataset,labels):
    separated = separateByClass(dataset,labels)
    prob_list = []
    labelsData = []
    datasetSize = len(labels)
    for classValue, instances in separated.items():
        pixelValues = []
        label = classValue
        labelSize = len(instances)
        pixelValues = [attribute for attribute in zip(*instances)]
        for i in range(0,len(pixelValues)):
            count_1 = 0.0
            for j in range(0,len(pixelValues[i])):
                if (pixelValues[i][j]==1):
                    count_1 +=1.0
            conditional_prob = (count_1/float(labelSize))
            prob_list.append([label,i,conditional_prob])
        labelsData.append([classValue,float(labelSize)/float(datasetSize)])
    return (prob_list,labelsData)

def separateByClass(dataset,labels):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (labels[i] not in separated):
            separated[labels[i]] = []
        separated[labels[i]].append(vector)
    return separated

def binarization(delta,array):
    return (np.where(array>delta, 1, 0))

def predictLabels(POS_LABEL,prob_list,testLabels):
    labelsProbability = []
    c = 0
    print (len(testLabels))
    for testLabel in testLabels:
        probability = 1.0
        for i in range(0,len(testLabel)):
            probability *= probCal(testLabel[i],POS_LABEL,prob_list,i)
        print (c)
        c+=1
        labelsProbability.append(probability)
    return labelsProbability

def probCal(pixel,POS_LABEL,prob_list,i):
    for a in prob_list:
        if(a[0]==POS_LABEL and a[1]==i):
            if(a[2]==0.0):
                return 1.0
            else:
                if(pixel == 0):
                    return 1.0-a[2]
                else:
                    return a[2]

def Binarythreshold(c1,c2,threshold,x):
    mylist = []
    for a in x:
        if a>=threshold:
            mylist.append(c1)
        else:
            mylist.append(c2)
    return mylist

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def getResults(test_labels,output,C1,C2):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    for i in range(len(test_labels)):
        if(test_labels[i]==C1 and output[i]==C1):
            TP+=1.0
        elif(test_labels[i]==C2 and output[i]==C1):
            FP+=1.0
        elif(test_labels[i]==C1 and output[i]==C2):
            FN+=1.0
        elif(test_labels[i]==C2 and output[i]==C2):
            TN+=1.0

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    #return (TP,FP,FN,TN,FPR,TPR)
    return (FPR,TPR)

def confusionmatrix(actual, predicted, normalize = False):
    unique = sorted(set(actual))
    matrix = [[0 for _ in unique] for _ in unique]
    imap   = {key: i for i, key in enumerate(unique)}

    K = len(np.unique(actual))
    result = np.zeros((K, K))
    for p, a in zip(predicted, actual):
        result[imap[p]][imap[a]] += 1
    '''
    if normalize:
        sigma = sum([sum(matrix[imap[i]]) for i in unique])
        matrix = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
    '''
    return result

def precision(TP,FP):
    return (TP)/(TP + FP)

def recall(TP,FN):
    return (TP)/(TP + FN)

def get_confusion_matrix_values(cm):
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

def kfold(num_folds,X_train,y_train,X_test,y_test):
    subset_size = len(y_train)/num_folds
    for i in range(num_folds):
        testing_this_round = training[i*subset_size:][:subset_size]
        training_this_round = training[:i*subset_size] + training[(i+1)*subset_size:]

def k_fold_generator(X, y, k_fold):
    subset_size = int(len(X) / k_fold)  # Cast to int if using Python 3
    for k in range(k_fold):
        X_train_k = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_test_k = X[k * subset_size:][:subset_size]
        y_train_k = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_test_k = y[k * subset_size:][:subset_size]
        yield X_train_k, y_train_k, X_test_k, y_test_k

def main():
    #Variables
    POS = 1
    NEG = 8
    lw = 2
    num_folds = 5

    #Dataset
    X_train, y_train = load_mnist('data/mnist', kind='train')
    X_test, y_test = load_mnist('data/mnist', kind='t10k')
    X_train = binarization(DELTA,X_train)
    X_test = binarization(DELTA,X_test)
    X_train, y_train = classSeparator(POS,NEG,X_train,y_train)
    X_test, y_test = classSeparator(POS,NEG,X_test,y_test)

    #K-fold
    subset_size = len(y_train)/num_folds
    for X_train_k, y_train_k, X_test_k, y_test_k in k_fold_generator(X_train, y_train, num_folds):
        max = 0.0
        maxdict = {}
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        FPRList = []
        TPRList = []
        threshold = ["0.1e-"+str(i) for i in range(0,350)]
        prob = 0.0

        #X_train_k = X_train[i*subset_size:][:subset_size]
        #y_train_k = y_train[i*subset_size:][:subset_size]
        #X_test_k = X_train[:i*subset_size] + X_train[(i+1)*subset_size:]
        #y_test_k = y_train[:i*subset_size] + y_train[(i+1)*subset_size:]

        #Training
        prob_list,labels = dictmaker(X_train_k,y_train_k)
        predictions = predictLabels(POS,prob_list,X_test_k)
        pickleUnload("Predictions/Validation/"+str(i)+"_"+str(POS)+"_"+str(NEG)+"_predictions.pkl",predictions)
        predictions = pickleLoad("Predictions/Validation/"+str(i)+"_"+str(POS)+"_"+str(NEG)+"_predictions.pkl")
        print (i)

        for a in labels:
            if a[0]==POS:
                prob = a[1]

        #Results
        for i in range(0,len(threshold)):
            y_pred = Binarythreshold(POS,NEG,float(threshold[i]),[x*prob for x in predictions])
            accuracy = round(getAccuracy(y_test_k, y_pred),2)
            if(max<accuracy):
                confusion_matrix = confusionmatrix(y_test_k, y_pred)
                TP,FP,FN,TN = get_confusion_matrix_values(confusion_matrix)
                maxdict = {'Accuracy':accuracy,
                        'Precision':round(precision(TP,FP)*100,2),
                        'Recall':round(recall(TP,FN)*100,2),
                        'Confusion_Matrix':confusion_matrix}
                max = accuracy
            FPR,TPR = getResults(y_test_k,y_pred,POS,NEG)
            FPRList.append(FPR)
            TPRList.append(TPR)

        #Best Accuracy, Precision, Recall and Confusion Matrix
        eer = brentq(lambda x : 1. - x - interp1d(FPRList, TPRList)(x), 0., 1.)
        print ("Accuracy: {}%".format(maxdict['Accuracy']))
        print ("Precision: {}%".format(maxdict['Precision']))
        print ("Recall: {}%".format(maxdict['Recall']))
        print ("Equal Error Rate: {}%".format(round(eer*100.0,2)))
        print ("Confusion matrix: ")
        print(maxdict['Confusion_Matrix'])

    '''
    #ROC Curve Plot
    plt.plot(FPRList,TPRList,color='darkorange',lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()
    '''
main()
