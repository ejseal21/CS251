import classifiers
import sys
import data
import numpy as np
import datetime
def main(argv):
    time = datetime.datetime.now()
    # test function here
    if len(argv) < 4 or (argv[3] != 'k' and argv[3] != 'n') :
        print( 'Usage: python %s <training data file> <test data file> <n for Naive Bayes, k for KNN> <optional training categories file> <optional test categories file>' % (argv[0]) )
        print( '    If categories are not provided as separate files, then the last column is assumed to be the category.')
        exit(-1)
    
    train_file = argv[1]
    test_file = argv[2]
    knn = True if argv[3] == 'k' else False
    dtrain = data.Data(train_file)
    dtest = data.Data(test_file)
    
    if len(argv) >= 6:
        train_headers = dtrain.get_headers()
        test_headers = dtrain.get_headers()
        
        traincat_file = argv[4]
        testcat_file = argv[5]

        traincats = data.Data(traincat_file)
        traincatdata = traincats.limit_columns(traincats.get_headers())

        testcats = data.Data(testcat_file)
        testcatdata = testcats.limit_columns(testcats.get_headers())

    else:
        train_headers = dtrain.get_headers()[:-1]
        test_headers = dtrain.get_headers()[:-1]

        traincatdata = dtrain.limit_columns([dtrain.get_headers()[-1]])
        testcatdata = dtest.limit_columns([dtest.get_headers()[-1]])

        uniquelabels, correctedtraincats = np.unique( traincatdata.T.tolist()[0], return_inverse = True)
        correctedtraincats = np.matrix([correctedtraincats]).T

        uniquelabels, correctedtestcats = np.unique( testcatdata.T.tolist()[0], return_inverse = True)
        correctedtestcats = np.matrix([correctedtestcats]).T

    
    if not knn:
        nbc = classifiers.NaiveBayes(dtrain, train_headers, traincatdata )

        print( 'Naive Bayes Training Set Results' )
        A = dtrain.limit_columns(train_headers)
        
        newcats, newlabels = nbc.classify( A )
        traincats = newcats
               
        print('making confusion matrix')
        confmtx = nbc.confusion_matrix( correctedtraincats, newcats )


        print( nbc.confusion_matrix_str( confmtx ) )

        
        print( 'Naive Bayes Test Set Results' )
        for i in range(len(test_headers)):
            try:
                test_headers[i] = int(test_headers[i])
            except:
                break

        A = dtest.limit_columns(test_headers)
        
        print('classifying with naive bayes classifier')
        newcats, newlabels = nbc.classify( A )


        print('confusion matrix')
        confmtx = nbc.confusion_matrix( correctedtestcats, newcats )
        print( nbc.confusion_matrix_str( confmtx ) )
        
    else:
        print('knn')
        print( '-----------------' )
        print( 'Building KNN Classifier' )
        knnc = classifiers.KNN( dtrain, train_headers, traincatdata, 3 )

        print( 'KNN Training Set Results' )
        A = dtrain.limit_columns(train_headers)

        newcats, newlabels = knnc.classify( A )
        traincats = newcats
        confmtx = knnc.confusion_matrix( correctedtraincats, newcats )
        print( knnc.confusion_matrix_str(confmtx) )

        print( 'KNN Test Set Results' )
        A = dtest.limit_columns(test_headers)

        newcats, newlabels = knnc.classify(A)

        print('KNN TEST::Correct labels\n', correctedtestcats.T)
        print('KNN TEST:::Predicted labels\n', newcats)

        # print the confusion matrix
        confmtx = knnc.confusion_matrix( correctedtestcats, newcats )
        print( knnc.confusion_matrix_str(confmtx) )

    test_headers.append('predicted')
    
    dtest.add_header2col('predicted')
    dtest.add_column(newcats.T)
    dtest.write("heresyourdata.csv", test_headers)
    return
    
if __name__ == "__main__":
    main(sys.argv)