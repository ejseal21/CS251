# Evan Arthur
# Spring 2019
# CS 251 Project 8


import sys
import data
import analysis as an
import numpy as np
import classifiers
import csv

def classify(trainingSet, testSet,bayes = True, optrainingCats = None, optestCats = None,outputFile = "KNN.csv"):
    print("in classify")
    dtrain = data.Data(trainingSet)
    dtest = data.Data(testSet)
    if optrainingCats != None:
        trainHeaders = dtrain.get_headers()
        trainCats = data.Data(optrainingCats)
        trainCatsData = trainCats.newMatrix(trainCats.get_headers())
    else:
        trainHeaders = dtrain.get_headers()[:-1]
        trainCatsData = dtrain.newMatrix([dtrain.get_headers()[-1]])

    if optestCats != None:
        testHeaders = dtrain.get_headers()
        testCats = data.Data(optestCats)
        testCatsData = testCats.newMatrix(testCats.get_headers())
    else:
        testHeaders = dtrain.get_headers()[:-1]
        testCatsData = dtest.newMatrix([dtest.get_headers()[-1]])

    if bayes:
        nbc = classifiers.NaiveBayes(dtrain, trainHeaders, trainCatsData)

        print('Naive Bayes Training Set Results')
        A = dtrain.newMatrix(trainHeaders)

        newcats, newlabels = nbc.classify(A)

        uniquelabels, correctedtraincats = np.unique(trainCatsData.T.tolist()[0], return_inverse=True)
        correctedtraincats = np.matrix([correctedtraincats]).T

        confmtx = nbc.confusion_matrix(correctedtraincats, newcats)
        print(nbc.confusion_matrix_str(confmtx))
        print('Naive Bayes Test Set Results')
        A = dtest.newMatrix(testHeaders)

        newcats, newlabels = nbc.classify(A)
        uniquelabels, correctedtestcats = np.unique(testCatsData.T.tolist()[0], return_inverse=True)
        correctedtestcats = np.matrix([correctedtestcats]).T

        confmtx = nbc.confusion_matrix(correctedtestcats, newcats)
        print(nbc.confusion_matrix_str(confmtx))

        with open(outputFile,mode='w') as file:
            dataToWrite = A.tolist()
            writer = csv.writer(file)
            testHeaders.append("predicted categories")
            writer.writerow(testHeaders)
            writer.writerow(["numeric" for i in range(len(testHeaders))])
            for i in range(len(dataToWrite)):
                dataToWrite[i].append(newcats[i,0])
                writer.writerow(dataToWrite[i])


    else:
        print('Building KNN Classifier')
        knnc = classifiers.KNN(dtrain, trainHeaders, trainCatsData, 5)

        print('KNN Training Set Results')
        A = dtrain.newMatrix(trainHeaders)

        newcats, newlabels = knnc.classify(A)
        uniquelabels, correctedtraincats = np.unique(trainCatsData.T.tolist()[0], return_inverse=True)
        correctedtraincats = np.matrix([correctedtraincats]).T

        confmtx = knnc.confusion_matrix(correctedtraincats, newcats)
        print(knnc.confusion_matrix_str(confmtx))

        print('KNN Test Set Results')
        A = dtest.newMatrix(testHeaders)

        newcats, newlabels = knnc.classify(A)

        uniquelabels, correctedtestcats = np.unique(testCatsData.T.tolist()[0], return_inverse=True)
        correctedtestcats = np.matrix([correctedtestcats]).T

        # print the confusion matrix
        confmtx = knnc.confusion_matrix(correctedtestcats, newcats)
        print(knnc.confusion_matrix_str(confmtx))

        with open(outputFile,mode='w') as file:
            dataToWrite = A.tolist()
            writer = csv.writer(file)
            testHeaders.append("predicted categories")
            writer.writerow(testHeaders)
            writer.writerow(["numeric" for i in range(len(testHeaders))])
            for i in range(len(dataToWrite)):
                dataToWrite[i].append(newcats[i,0])
                writer.writerow(dataToWrite[i])


def main(argv):
    # test function here
    if len(argv) < 4:
        print( 'Usage: python %s <training data file> <test data file> <bayes(t/f)> <optional training categories file> <optional test categories file>' % (argv[0]) )
        print( '    If categories are not provided as separate files, then the last column is assumed to be the category.')
        exit(-1)
    if argv[3] == 'T' or argv[3] == "t":
        bayes = True
    else:
        bayes = False
    if len(argv) == 4:
        print("here")
        print(argv[3])
        classify(argv[1],argv[2],bayes)
    elif len(argv) == 5:
        classify(argv[1],argv[2],bayes,argv[4])
    elif len(argv) == 6:
        classify(argv[1], argv[2],bayes, argv[4], argv[5])

if __name__ == "__main__":
    main(sys.argv)