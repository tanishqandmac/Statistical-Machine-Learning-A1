import csv
import random
import math
from utils import mnist_reader
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
import pickle
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
import matplotlib.pyplot as plt

DELTA = 127

def pickleLoad(filename):
    with open(filename, "rb") as f:
        filetype = pickle.load(f)
    return filetype

def pickleUnload(filename,filetype):
    with open(filename, "wb") as f:
        pickle.dump(filetype, f)

def binarization(delta,array):
    return (np.where(array>delta, 1, 0))

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	return summaries

def calculateProbability(x, mean, stdev):
    if(mean == 0 or stdev == 0):
        return 1
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def summarizeByClass(dataset,labels):
    separated = separateByClass(dataset,labels)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    #print (summaries)
    return summaries

def separateByClass(dataset,labels):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (labels[i] not in separated):
            separated[labels[i]] = []
        separated[labels[i]].append(vector)
    return separated

def precision(TP,FP):
    return (TP)/(TP + FP)

def recall(TP,FN):
    return (TP)/(TP + FN)

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

def get_confusion_matrix_values(cm):
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

def ROC(y_pred_proba,y_test):
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
    #plt.plot(score,y)
    #plt.show()

def classSeparator(r1,r2,X_train,y_train):
    X_train_re = []
    y_train_re = []
    for i in range(0,len(X_train)):
        if (y_train[i] == r1 or y_train[i] == r2):
            y_train_re.append(y_train[i])
            X_train_re.append(X_train[i])
    return (X_train_re, y_train_re)

def tptn(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return (TP,FP,FN,TN)

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
    return (TP,FP,FN,TN)

def main():
    X_test, y_test = mnist_reader.load_mnist('data/mnist', kind='t10k')
    X_test, y_test = classSeparator(1,8,X_test,y_test)
    k = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255]
    '''
    for i in range(len(k)):
        X_train, y_train = mnist_reader.load_mnist('data/mnist', kind='train')
        X_test, y_test = mnist_reader.load_mnist('data/mnist', kind='t10k')
        X_train = binarization(k[i],X_train)
        X_test = binarization(k[i],X_test)
        X_train, y_train = classSeparator(1,8,X_train,y_train)
        X_test, y_test = classSeparator(1,8,X_test,y_test)
        print ("Class Separated!")
        summaries = summarizeByClass(X_train,y_train)
        print ("Summaries Done!")
        pickleUnload("Scrap/nb_"+str(k[i])+"_model.pkl",summaries)
        summaries = pickleLoad("Scrap/nb_"+str(k[i])+"_model.pkl")
        predictions = getPredictions(summaries, X_test)
        print ("Predictions Done!")
        pickleUnload("Scrap/nb_"+str(k[i])+"_predictions.pkl",predictions)
    '''
    tpr = []
    fpr = []
    predictions = pickleLoad("Scrap/nb_"+str(3)+"_predictions.pkl")
    fpr, tpr, thresholds = roc_curve(y_test, predictions, pos_label='1')
    roc_auc = auc(y_test, predictions)
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    '''

    for i in range(len(k)):
        X_test, y_test = mnist_reader.load_mnist('data/mnist', kind='t10k')
        X_test, y_test = classSeparator(1,8,X_test,y_test)
        predictions = pickleLoad("Scrap/nb_"+str(k[i])+"_predictions.pkl")
        accuracy = getAccuracy(y_test, predictions)
        TP,FP,FN,TN = getResults(y_test, predictions,1,8)
        confusion_matrix = confusionmatrix(y_test,predictions)
        print('Accuracy: {}%'.format(round(accuracy,2)))
        print('Precision: {}%'.format(round(precision(TP,FP)*100,2)))
        print('Recall: {}%'.format(round(recall(TP,FN)*100,2)))
        print('Confusion Matrix:')
        print(confusion_matrix)
        #k += 10
        tpr.append(float(confusion_matrix[0][0])/(float(confusion_matrix[0][0])+float(confusion_matrix[1][0])))
        fpr.append(float(confusion_matrix[0][1])/(float(confusion_matrix[0][1])+float(confusion_matrix[1][1])))
    #print ("[{} {}]".format(int(TP),int(FP)))
    #print ("[{} {}]".format(int(FN),int(TN)))
    #ROC(predictions,y_test)
    #fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
    lw = 2
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()
    '''
'''
    skplt.metrics.plot_roc_curve(y_test, predictions)
    plt.show()
    print (y_test[:, 1])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
'''
main()
