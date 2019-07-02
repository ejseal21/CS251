# Evan Arthur
# Spring 2019
# CS 251 Project 8
#
# Classifier class and child definitions

import sys
import data
import analysis as an
import numpy as np

class Classifier:

    def __init__(self, type):
        '''The parent Classifier class stores only a single field: the type of
        the classifier.  A string makes the most sense.

        '''
        self._type = type
        self.numClasses = 0

    def type(self, newtype = None):
        '''Set or get the type with this function'''
        if newtype != None:
            self._type = newtype
        return self._type

    def confusion_matrix( self, truecats, classcats ):
        '''Takes in two Nx1 matrices of zero-index numeric categories and
        computes the confusion matrix. The rows represent true
        categories, and the columns represent the classifier output.
        To get the number of classes, you can use the np.unique
        function to identify the number of unique categories in the
        truecats matrix.

        '''
        temp = [[0 for i in range(self.numClasses)] for j in range(self.numClasses)]
        for i in range(len(truecats)):
            temp[truecats[i,0]][classcats[i,0]] += 1
        return temp

    def confusion_matrix_str( self, cmtx ):
        '''Takes in a confusion matrix and returns a string suitable for printing.'''
        s = ''
        for i in cmtx:
            s += "|"
            for j in i:
                s += ' ' + str(j) + " "
            s += "|\n"

        return s

    def __str__(self):
        '''Converts a classifier object to a string.  Prints out the type.'''
        return str(self._type)



class NaiveBayes(Classifier):
    '''NaiveBayes implements a simple NaiveBayes classifier using a
    Gaussian distribution as the pdf.

    '''

    def __init__(self, data=None, headers=[], categories=None):
        '''Takes in a Matrix of data with N points, a set of F headers, and a
        matrix of categories, one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'Naive Bayes Classifier')
        self.headers = headers
        # store the headers used for classification
        # number of classes and number of features
        # original class labels
        # unique data for the Naive Bayes: means, variances, scales, priors
        # if given data,
            # call the build function
        # print(data.get_data())
        if data != None:
            self.build(data.newMatrix(headers),categories)

    def build( self, A, categories ):
        '''Builds the classifier give the data points in A and the categories'''
        self.numCategories = len(categories)
        self.numFeatures = A.shape[1]
        self.unique, self.mapping,self.counts = np.unique(np.array(categories.T), return_inverse=True, return_counts= True)
        self.numClasses = self.unique.size
        self.ogLabels = self.unique
        # figure out how many categories there are and get the mapping (np.unique)
        self.classMeans = np.matrix(np.zeros((self.numClasses, self.numFeatures)))
        self.classVars = np.matrix(np.zeros((self.numClasses, self.numFeatures)))
        self.classScales = np.matrix(np.zeros((self.numClasses, self.numFeatures)))
        # create the matrices for the means, vars, and scales
        for i in range(self.numClasses):
            self.classMeans[i, :] = np.mean(A[(self.mapping == i), :], axis=0)
            self.classVars[i, :] = np.var(A[(self.mapping == i), :], axis=0, ddof=1)
            # print(self.classVars[i,:])
            self.classScales[i, :] = (1 / np.sqrt(2 * np.pi * self.classVars[i, :]))
        self.priors = []
        for count in self.counts:
            self.priors.append(float(count)/float(len(self.mapping)))
        # the output matrices will be categories x features
        # compute the means/vars/scales/priors for each class
        # the prior for class i will be the number of examples in class i divided by the total number of examples
        # store any other necessary information: # of classes, # of features, original labels

        return

    def classify( self, A, return_likelihoods=False ):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_likelihoods
        is True, it also returns the probability value for each class, which
        is product of the probability of the data given the class P(Data | Class)
        and the prior P(Class).

        '''
        # print("shape: ", A.shape)
        # error check to see if A has the same number of columns as the class means
        if(A.shape[1] != self.classMeans.shape[1]):
            print("not right size")
            return
        # make a matrix that is N x C to store the probability of each class for each data point
        # a matrix of zeros that is N (rows of A) x C (number of classes)
        P = np.matrix(np.zeros((A.shape[0], self.numClasses)))

        for i in range(self.numClasses):
            P[:,i] = np.prod(np.multiply(self.classScales[i,:],np.exp(-np.square(A-self.classMeans[i,:])/(2*self.classVars[i,:]))),axis=1) * self.priors[i]
        # Calcuate P(D | C) by looping over the classes
        #  with numpy-fu you can do this in one line inside a for
        #  loop, calculating a column of P in each loop.
        #  To compute the likelihood, use the formula for the Gaussian
        #  pdf for each feature, then multiply the likelihood for all
        #  the features together The result should be an N x 1 column
        #  matrix that gets assigned to a column of P

        # Multiply the likelihood for each class by its corresponding prior

        # calculate the most likely class for each data point
        # take the argmax of P along axis 1
        cats = np.argmax(P, axis=1)
        # use the class ID as a lookup to generate the original labels
        labels = self.ogLabels[cats]

        if return_likelihoods:
            return cats, labels, P

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nNaive Bayes Classifier\n"
        for i in range(self.numClasses):
            s += 'Class %d --------------------\n' % (i)
            s += 'Mean  : ' + str(self.classMeans[i,:]) + "\n"
            s += 'Var   : ' + str(self.classVars[i,:]) + "\n"
            s += 'Scales: ' + str(self.classScales[i,:]) + "\n"

        s += "\n"
        return s
        
    def write(self, filename):
        '''Writes the Bayes classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the Bayes classifier from the file'''
        # extension
        return

    
class KNN(Classifier):

    def __init__(self, data=None, headers=[], categories=None, K=None):
        '''Take in a Matrix of data with N points, a set of F headers, and a
        matrix of categories, with one category label for each data point.'''

        # call the parent init with the type
        Classifier.__init__(self, 'KNN Classifier')
        
        # store the headers used for classification
        self.headers = headers
        # number of classes and number of features
        # original class labels
        # unique data for the KNN classifier: list of exemplars (matrices)
        # if given data,
            # call the build function
        if data != None:
            self.build(data,categories,K)

    def build( self, A, categories, K = None ):
        '''Builds the classifier give the data points in A and the categories'''

        # figure out how many categories there are and get the mapping (np.unique)
        # print(A.get_data())
        self.numCategories = len(categories)
        self.numFeatures = A.get_data().shape[1]
        self.unique, self.mapping,self.counts = np.unique(np.array(categories.T), return_inverse=True, return_counts= True)
        self.numClasses = self.unique.size
        self.ogLabels = self.unique
        self.exemplars = [A.get_data()[(self.mapping == i),:] for i in range(self.numClasses)]
        if K != None:
            temp = []
            for e in self.exemplars:
                codebook, codes, error,quality = an.kmeans(None, headers=None, whiten=False, K=K, A=e)
                temp.append(codebook)
            self.exemplars = temp
        return

    def classify(self, A, return_distances=False, K=6):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matrix. The distance is 
        calculated using the nearest K neighbors.'''
        D = np.zeros((A.shape[0],len(self.ogLabels)))
        sums = []
        for j in self.exemplars:
            temp = np.hstack([np.matrix([[np.linalg.norm(A[k, :] - i)] for k in range(A.shape[0])]) for i in j])
            temp = np.sort(temp,axis=1)
            sum = np.sum(temp[:,:K],axis=1)
            sums.append(sum)
        D = np.hstack(sums)

        # calculate the most likely class for each data point
        # take the argmin of D along axis 1
        cats = np.argmin(D,axis=1)

        # use the class ID as a lookup to generate the original labels
        labels = self.ogLabels[cats]

        if return_distances:
            return cats, labels, D
        # print("cats: ", cats)
        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nKNN Classifier\n"
        for i in range(self.numClasses):
            s += 'Class %d --------------------\n' % (i)
            s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
            s += 'Mean of Exemplars  :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

        s += "\n"
        return s


    def write(self, filename):
        '''Writes the KNN classifier to a file.'''
        # extension
        return

    def read(self, filename):
        '''Reads in the KNN classifier from the file'''
        # extension
        return
    

# test function
def main(argv):
    # test function here
    if len(argv) < 3:
        print( 'Usage: python %s <training data file> <test data file><optional training categories file> <optional test categories file>' % (argv[0]) )
        print( '    If categories are not provided as separate files, then the last column is assumed to be the category.')
        exit(-1)

    train_file = argv[1]
    test_file = argv[2]
    dtrain = data.Data(train_file)
    dtest = data.Data(test_file)


    if len(argv) >= 5:
        train_headers = dtrain.get_headers()
        test_headers = dtrain.get_headers()
        
        traincat_file = argv[3]
        testcat_file = argv[4]

        traincats = data.Data(traincat_file)
        traincatdata = traincats.newMatrix(traincats.get_headers())

        testcats = data.Data(testcat_file)
        testcatdata = testcats.newMatrix(testcats.get_headers())

    else:
        train_headers = dtrain.get_headers()[:-1]
        test_headers = dtrain.get_headers()[:-1]

        traincatdata = dtrain.newMatrix([dtrain.get_headers()[-1]])
        testcatdata = dtest.newMatrix([dtest.get_headers()[-1]])

    
    nbc = NaiveBayes(dtrain, train_headers, traincatdata )

    print( 'Naive Bayes Training Set Results' )
    A = dtrain.newMatrix(train_headers)
    
    newcats, newlabels = nbc.classify( A )

    uniquelabels, correctedtraincats = np.unique( traincatdata.T.tolist()[0], return_inverse = True)
    correctedtraincats = np.matrix([correctedtraincats]).T

    confmtx = nbc.confusion_matrix( correctedtraincats, newcats )
    print( nbc.confusion_matrix_str( confmtx ) )


    print( 'Naive Bayes Test Set Results' )
    A = dtest.newMatrix(test_headers)
    
    newcats, newlabels = nbc.classify( A )

    uniquelabels, correctedtestcats = np.unique( testcatdata.T.tolist()[0], return_inverse = True)
    correctedtestcats = np.matrix([correctedtestcats]).T

    confmtx = nbc.confusion_matrix( correctedtestcats, newcats )
    print( nbc.confusion_matrix_str( confmtx ) )

    print( '-----------------' )
    print( 'Building KNN Classifier' )
    knnc = KNN( dtrain, train_headers, traincatdata, 10 )

    print( 'KNN Training Set Results' )
    A = dtrain.newMatrix(train_headers)

    newcats, newlabels = knnc.classify( A )

    confmtx = knnc.confusion_matrix( correctedtraincats, newcats )
    print( knnc.confusion_matrix_str(confmtx) )

    print( 'KNN Test Set Results' )
    A = dtest.newMatrix(test_headers)

    newcats, newlabels = knnc.classify(A)

    # print the confusion matrix
    confmtx = knnc.confusion_matrix( correctedtestcats, newcats )
    print( knnc.confusion_matrix_str(confmtx) )

    return
    
if __name__ == "__main__":
    main(sys.argv)
