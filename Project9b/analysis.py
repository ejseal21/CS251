#Evan Arthur
#CS251
#2/24/19

import numpy as np
import data
import scipy.stats as stats
import scipy.cluster.vq as vq
import random

def data_range(colHeaders, data):
    #finds the rang of each column
    mat = data.newMatrix(colHeaders)
    max = np.max(mat,axis=0)
    max = max.tolist()
    max = max[0]

    min = np.min(mat,axis=0)
    min = min.tolist()
    min = min[0]

    result = np.array([max,min])
    result = result.transpose().tolist()
    return result

def mean(colHeaders, data):
    #returns the mean of the columns
    return np.mean(data.newMatrix(colHeaders),axis=0).tolist()

def stdev(colHeaders, data):
    #returns standard deviation of the columns
    return np.std(data.newMatrix(colHeaders),axis=0).tolist()

def normalize_columns_separately(colHeaders,data):
    #normalizes the columns seperately and returns the result
    mat = data.newMatrix(colHeaders)
    return (mat - mat.min(0)) / mat.ptp(0)

def normalize_columns_together(colHeaders,data):
    #normalizes the columns together and returns the resulting matrix
    mat = data.newMatrix(colHeaders)
    min, max = mat.min(), mat.max()
    mat = (mat - min)/(max - min)
    return mat

def single_linear_regression(data_obj, ind_var, dep_var):
    mat = data_obj.newMatrix([ind_var,dep_var])
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(np.asarray(mat[:,0])),np.squeeze(np.asarray(mat[:,1])))
    indRange = data_range([ind_var],data_obj)
    depRange = data_range([dep_var],data_obj)
    return (slope, intercept, r_value, p_value, std_err, indRange[0][0], indRange[0][1], depRange[0][0], depRange[0][1])

def linear_regression(d,ind,dep):
    y = d.newMatrix([dep])
    a = d.newMatrix(ind)
    a = np.hstack((a,np.ones((np.size(a,0),1))))
    aainv = np.linalg.inv( np.dot(a.T, a))
    x = np.linalg.lstsq( a, y,rcond=None)
    b = x[0]
    n = np.size(y,0)
    c = np.size(b,0)
    df_e = n-c
    df_r = c-1
    error = y - np.dot(a, b)
    sse = np.dot(error.T, error) / df_e
    stderr = np.sqrt( np.diagonal( sse[0, 0] * aainv ) )
    t = b.T / stderr
    p = 2*(1 - stats.t.cdf(abs(t), df_e))
    r2 = 1 - error.var() / y.var()
    return (b,sse,r2,t,p)

#using SVD
def pca(d, headers, prenormalize=True):
    if prenormalize:
        A = normalize_columns_separately(headers, d)
    else:
        A = d.limit_columns(headers)

    #do the mean on A so that it's normalized if they wanted to normalize it
    m = np.mean(A, axis=0)
    D = A - m
    U,S,V = np.linalg.svd(D, full_matrices=False)
    
    #D*V.T are the eigenvectors
    #V are the eigenvalues
    return data.PCAData(D*V.T,V,((S**2)/(A.shape[0]-1)),m,headers)


def kmeans_numpy(A, headers, K, whiten=True):
    '''Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes, and representation error.
    '''
    print("K: ", K)
    # assign to A the result of getting the data from your Data object
    # assign to W the result of calling vq.whiten on A
    W = vq.whiten(A)
    # assign to codebook, bookerror the result of calling vq.kmeans with W and K
    codebook,bookerror = vq.kmeans(W,K)
    # assign to codes, error the result of calling vq.vq with W and the codebook
    codes, error = vq.vq(W,codebook)
    # return codebook, codes, and error
    return codebook,codes,error

# Selects K random rows from the data matrix A and returns them as a matrix
def kmeans_init(A, K):
    pts = A.shape[0]
    if K > pts:
        print("not enough data points")
        return
    indices = []
    for i in range(K):
        index = random.randint(0,pts - 1)
        while index in indices:
            index = random.randint(0,pts-1)
        indices.append(index)
    return A[indices,:]
    # Hint: generate a list of indices then shuffle it and pick K
    # return np.matrix(random.shuffle(list(range(0,np.size(A,0))))[:K])
    # Hint: Probably want to check for error cases (e.g. # data points < K)


# Given a data matrix A and a set of means in the codebook
# Returns a matrix of the id of the closest mean to each point
# Returns a matrix of the sum-squared distance between the closest mean and each point
def kmeans_classify(A, codebook, manhattan=False):
    # Hint: you can compute the difference between all of the means and data point i using: codebook - A[i,:]
    roots = []
    indices = []
    distances = []
    for i in range(len(A)):
        temp_dist = []
        for j in range(len(codebook)):
            if not manhattan:
                # print("here")
                temp_dist.append(np.sqrt(np.sum(np.square(codebook[j,:] - A[i,:]))))
            else:
                temp_dist.append(np.sum(codebook[j,:] - A[i,:]))
        indices.append(temp_dist.index(min(temp_dist)))
        distances.append(min(temp_dist))
    return np.matrix(indices).T, np.matrix(distances).T


    # Hint: look at the numpy functions square and sqrt
    # Hint: look at the numpy functions argmin and min

# Given a data matrix A and a set of K initial means, compute the optimal
# cluster means for the data and an ID and an error for each data point
def kmeans_algorithm(A, means,MIN_CHANGE = 1e-7,MAX_ITERATIONS = 100,manhattan = False):
    # set up some useful constants
    D = means.shape[1]    # number of dimensions
    K = means.shape[0]    # number of clusters
    N = A.shape[0]        # number of data points

    # iterate no more than MAX_ITERATIONS
    for i in range(MAX_ITERATIONS):

        # calculate the codes by calling kemans_classify
        codes,distances = kmeans_classify(A,means,manhattan)
        # codes[j,0] is the id of the closest mean to point j
        # initialize newmeans to a zero matrix identical in size to means
        newmeans = np.zeros_like(means)
        # Hint: look up the numpy function zeros_like
        # Meaning: the new means given the cluster ids for each point

        # initialize a K x 1 matrix counts to zeros
        # Hint: use the numpy zeros function
        # Meaning: counts will store how many points get assigned to each mean
        counts = np.zeros((K,1))

        # for the number of data points
        for j in range(N):
            # add to the closest mean (row codes[j,0] of newmeans) the jth row of A
            newmeans[codes[j],:] += A[j,:]
            # add one to the corresponding count for the closest mean
            counts[codes[j],0] += 1

        # finish calculating the means, taking into account possible zero counts
        #for the number of clusters K
        for j in range(K):
            # if counts is not zero, divide the mean by its count
            if counts[j,0] != 0:
                newmeans[j,:] /= counts[j,0]
            # else pick a random data point to be the new cluster mean
            else:
                newmeans[j,:] = A[random.randint(0,N),:]

        # test if the change is small enough and exit if it is
        diff = np.sum(np.square(means - newmeans))
        means = newmeans
        if diff < MIN_CHANGE:
            break

    # call kmeans_classify one more time with the final means
    codes, errors = kmeans_classify( A, means )

    # return the means, codes, and errors
    return (means, codes, errors)


def kmeans(d, headers = [], K = 3, whiten=True,manhattanh = False,A=None):
    '''Takes in a Data object, a set of headers, and the number of clusters to create
    Computes and returns the codebook, codes and representation errors.
    '''
    for i in range(2):
        try:
            A = d.newMatrix(headers)
            d.set_kmeans()
        except:
            pass
        if whiten:
            W = vq.whiten(A)
        else:
            W = A

    # assign to A the result getting the data given the headers
    # if whiten is True
    # print("w: ", W)
    # assign to codebook the result of calling kmeans_init with W and K
    codebook = kmeans_init(W,K)
    # print("got codebook")
    # assign to codebook, codes, errors, the result of calling kmeans_algorithm with W and codebook

    codebook, codes, errors = kmeans_algorithm(W, codebook, manhattanh)
    # print("got through alg")
    quality = kmeans_quality(errors,K)
    # return the codebook, codes, and representation error
    return codebook,codes,errors,quality

def kmeans_quality(errors, K):
    return np.sum(np.square(errors)) + (K/2) * np.log2(errors.shape[0])

if __name__ == '__main__':
    data = data.Data(filename="data-noisy.csv")
    results = linear_regression(data,["X0","X1"],"Y")
    print("m0: ",results[0][0,0])
    print("m1: ",results[0][1,0])
    print("b: ",results[0][2,0])
    print("sse: ",results[1][0,0])
    print("R2: ",results[2])
    print("t: ",results[3])
    print("p: ",results[4])