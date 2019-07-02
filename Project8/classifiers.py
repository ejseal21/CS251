# Template by Bruce Maxwell
# Spring 2015
# CS 251 Project 8
#
# Classifier class and child definitions

import sys
import data
import analysis
import numpy as np
import scipy.cluster.vq as vq

class Classifier:

    def __init__(self, type):
        '''The parent Classifier class stores only a single field: the type of
        the classifier.  A string makes the most sense.

        '''
        self._type = type
        self.exemplars = []

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
        if truecats.shape != classcats.shape:
            classcats = classcats.T
        unique, mapping = np.unique(np.array(truecats), return_inverse = True)
        self.num_classes = len(unique)
        confmtx = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for i in range(self.num_classes):
            row = classcats[(mapping==i),:]
            for j in range(len(row)):
                confmtx[i,row[j,0]] += 1
            # confmtx[truecats[i,0],classcats[i,0]] += 1
        return confmtx

    def confusion_matrix_str( self, cmtx ):
        '''Takes in a confusion matrix and returns a string suitable for printing.'''
        s = 'Confusion Matrix:\n'
        s += '\tClassified As\n'
        s +=  '\nTruth\n\t'

        for i in range(self.num_classes):
            s += str(i) + '\t'
        
        s += '\n'
        size = cmtx.shape[0]
        for i in range(cmtx.shape[0]):
            s+= '\n'
            s += str(i)
            for j in range(cmtx.shape[1]):
                
                s += '\t'
                s+= str(cmtx[i,j])
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
        # store the headers used for classification

        # if given data,
            # call the build function
        if data != None:
            self.build(data, categories)
        self.headers= headers
        # number of classes and number of features

    def build( self, A, categories ):
        '''Builds the classifier give the data points in A and the categories'''
        if isinstance(A,data.Data):
            A = A.get_data()
        self.num_cats = len(categories)
        self.num_feats = A.shape[1]
        self.num_points = A.shape[0]
        # figure out how many categories there are and get the mapping (np.unique)
        self.unique, self.mapping, self.counts = np.unique( np.array(categories.T), return_inverse=True, return_counts=True)
        self.class_labels = self.unique
        self.num_classes = len(self.unique)

        # create the matrices for the means, vars, and scales
        # the output matrices will be categories x features
        self.class_means = np.matrix(np.zeros((self.num_classes, self.num_feats)))
        self.class_vars = np.matrix(np.zeros((self.num_classes, self.num_feats)))
        self.class_scales = np.matrix(np.zeros((self.num_classes, self.num_feats)))

        
        # compute the means/vars/scales/priors for each class
        for i in range(len(self.unique)):
            self.class_means[i,:] = np.mean(A[(self.mapping==i), :], axis=0)
            self.class_vars[i,:] = np.var(A[(self.mapping==i), :], axis=0, ddof=1)
            # i have to add 0.1 to the denominator so that it's not a divide by 0 error
            self.class_scales[i,:] = 1/(0.1 + np.sqrt(2*np.pi*self.class_vars[i,:]))
        
        # the prior for class i will be the number of examples in class i divided by the total number of examples
        self.priors = []
        for i in range(len(self.counts)):
            self.priors.append(self.counts[i]/self.num_points)


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

        # error check to see if A has the same number of columns as the class means
        print(A.shape[1], self.class_means.shape[1])
        while A.shape[1] != self.class_means.shape[1]:
            self.class_means = self.class_means[:, 0:self.class_means.shape[1]-1]
            self.class_vars = self.class_vars[:, 0:self.class_vars.shape[1]-1]
            self.class_scales = self.class_scales[:, 0:self.class_scales.shape[1]-1]
            if self.class_means.shape[1] == 0:
                return

        # make a matrix that is N x C to store the probability of each class for each data point
        P = np.matrix(np.zeros((A.shape[0], self.num_classes))) # a matrix of zeros that is N (rows of A) x C (number of classes)
        
        # Calcuate P(D | C) by looping over the classes
        #  with numpy-fu you can do this in one line inside a for
        #  loop, calculating a column of P in each loop.
        #
        #  To compute the likelihood, use the formula for the Gaussian
        #  pdf for each feature, then multiply the likelihood for all
        #  the features together The result should be an N x 1 column
        #  matrix that gets assigned to a column of P
        # square, exp, multiply, prod
        for i in range(self.num_classes):
            #this does all the math
            P[:,i] = np.prod(np.multiply(self.class_scales[i,:], np.exp(-np.square(A - self.class_means[i,:])/(2*self.class_vars[i,:]))), axis=1) * self.priors[i]
        

        # Multiply the likelihood for each class by its corresponding prior

        # calculate the most likely class for each data point
        cats = np.matrix(np.argmax(P,axis=1)).T # take the argmax of P along axis 1
        
        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_likelihoods:
            return cats, labels, P

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nNaive Bayes Classifier\n"
        for i in range(self.num_classes):
            s += 'Class %d --------------------\n' % (i)
            s += 'Mean  : ' + str(self.class_means[i,:]) + "\n"
            s += 'Var   : ' + str(self.class_vars[i,:]) + "\n"
            s += 'Scales: ' + str(self.class_scales[i,:]) + "\n"

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
        # if given data, call the build function
        if data != None:
            self.build(data, categories)

    def build( self, A, categories, K = None ):
        '''Builds the classifier give the data points in A and the categories'''
        try: 
            A = A.get_data()
        except:
            pass
        self.num_cats = len(categories)
        self.num_feats = A.shape[1]-1
        self.num_points = A.shape[0]

        # figure out how many categories there are and get the mapping (np.unique)
        self.unique, self.mapping, self.counts = np.unique( np.array(categories.T), return_inverse=True, return_counts=True)
        self.class_labels = self.unique
        self.num_classes = self.unique.size
        self.class_means = np.matrix(np.zeros((self.num_classes, self.num_feats)))
        self.class_vars = np.matrix(np.zeros((self.num_classes, self.num_feats)))
        self.class_scales = np.matrix(np.zeros((self.num_classes, self.num_feats)))
        self.exemplars = [A[(self.mapping==i),:-1] for i in range(self.num_classes)]
        if K != None:
            temp = []
            for i in range(len(self.exemplars)):    
                codebook, bookerror = vq.kmeans(A[(self.mapping==i),:], K)
                temp.append(codebook)
            self.exemplars = temp
            
        # store any other necessary information: # of classes, # of features, original labels

        return

    def classify(self, A, return_distances=False, K=3):
        '''Classify each row of A into one category. Return a matrix of
        category IDs in the range [0..C-1], and an array of class
        labels using the original label values. If return_distances is
        True, it also returns the NxC distance matrix. The distance is 
        calculated using the nearest K neighbors.'''

        # make a matrix that is N x C to store the distance to each class for each TEST data point
        D = np.matrix(np.zeros((A.shape[0], self.num_classes))) # a matrix of zeros that is N (rows of A) x C (number of classes)
        sums = []
        for i in range (self.num_classes):
            # N num test points of class i x Number of training exemplars of class i
            dist = np.matrix(np.zeros((A.shape[0], self.exemplars[i].shape[0]))) # 105x35
            for m in range(A.shape[0]):
                # for the m-th test point, we want to calculate the distance to all exemplars
                # Training exemplars. We want distance to all these
                # This is distance from test point M to ALL training exemplars of class i. 35x4
                dist[m,:] = np.sqrt( np.sum(np.square(A[m, :] - self.exemplars[i]), 1)).T
            #105x35
            dist = np.sort( dist, axis=1 )
            dist = np.asmatrix(dist)
            # Want to be 1x105 in size
            D[:,i] = np.sum(dist[:,:K],axis=1)
            
        # calculate the most likely class for each data point
        cats = np.matrix(np.argmin(D, axis =1)).T # take the argmin of D along axis 1

        # use the class ID as a lookup to generate the original labels
        labels = self.class_labels[cats]

        if return_distances:
            return cats, labels, D

        return cats, labels

    def __str__(self):
        '''Make a pretty string that prints out the classifier information.'''
        s = "\nKNN Classifier\n"
        for i in range(self.num_classes):
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
    
def main(argv):
    # test function here
    if len(argv) < 3:
        print( 'Usage: python %s <training data file> <test data file> <optional training categories file> <optional test categories file>' % (argv[0]) )
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
        traincatdata = traincats.limit_columns(traincats.get_headers())

        testcats = data.Data(testcat_file)
        testcatdata = testcats.limit_columns(testcats.get_headers())

    else:
        train_headers = dtrain.get_headers()[:-1]
        test_headers = dtrain.get_headers()[:-1]

        traincatdata = dtrain.limit_columns([dtrain.get_headers()[-1]])
        testcatdata = dtest.limit_columns([dtest.get_headers()[-1]])

    
    nbc = NaiveBayes(dtrain, train_headers, traincatdata )

    print( 'Naive Bayes Training Set Results' )
    A = dtrain.limit_columns(train_headers)
    newcats, newlabels = nbc.classify( A )
    uniquelabels, correctedtraincats = np.unique( traincatdata.T.tolist()[0], return_inverse = True)
    
    correctedtraincats = np.matrix([correctedtraincats]).T
    
    confmtx = nbc.confusion_matrix( correctedtraincats, newcats )
    print( nbc.confusion_matrix_str( confmtx ) )


    print( 'Naive Bayes Test Set Results' )
    print('test_headers', test_headers)
    for i in range(len(test_headers)):
        try:
            test_headers[i] = int(test_headers[i])
        except:
            break
    A = dtest.limit_columns(test_headers)
    
    newcats, newlabels = nbc.classify( A )

    uniquelabels, correctedtestcats = np.unique( testcatdata.T.tolist()[0], return_inverse = True)
    correctedtestcats = np.matrix([correctedtestcats]).T

    confmtx = nbc.confusion_matrix( correctedtestcats, newcats )
    print( nbc.confusion_matrix_str( confmtx ) )

    print( '-----------------' )
    print( 'Building KNN Classifier' )
    knnc = KNN( dtrain, train_headers, traincatdata, 10 )

    print( 'KNN Training Set Results' )
    A = dtrain.limit_columns(train_headers)

    newcats, newlabels = knnc.classify( A )

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

    return
    
if __name__ == "__main__":
    main(sys.argv)