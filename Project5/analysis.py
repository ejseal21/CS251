import numpy as np
import data
import scipy.stats
import sys

def single_linear_regression(data_obj, ind_var, dep_var):
	new_data = data_obj.limit_columns([ind_var, dep_var])
	linres = scipy.stats.linregress(new_data)

	#wow look at how many variables I can assign
	slope = linres[0]
	intercept = linres[1]
	rvalue = linres[2]
	pvalue = linres[3]
	stderr = linres[4]
	ind_min = data_range([ind_var],data_obj)[0][0]
	ind_max = data_range([ind_var], data_obj)[0][1]
	dep_min = data_range([dep_var],data_obj)[0][0]
	dep_max = data_range([dep_var], data_obj)[0][1]
	#9 variables, one after the other? Wow!

	return (slope, intercept, rvalue, pvalue, stderr, ind_min, ind_max, dep_min, dep_max)

def linear_regression(d, ind, dep):
	mat = d.limit_columns([*ind, dep])
	y = mat[:,2]
	A = mat[:,0:2]
	print(A)
	A = np.hstack((A, np.ones((mat.shape[0], 1))))
	AAinv = np.linalg.inv(np.dot(A.T, A))
	x = np.linalg.lstsq(A,y)
	b = x[0]
	N = y.shape[0]
	C = b.shape[0]
	df_e = N-C
	df_r = C-1
	error = y - np.dot(A,b)
	sse = np.dot(error.T, error) / df_e
	stderr = np.sqrt(np.diagonal(sse[0,0] * AAinv))
	t = b.T / stderr
	p = 2*(1 - scipy.stats.t.cdf(abs(t),df_e))
	r2 = 1 - error.var() / y.var()

	return b[0],b[1],b[2],sse, r2, t, p

#Extension 5
#return a 2d array with minimum value-maximum value for each column

def data_range(headers, dobj, combine = False):
	#make a matrix of just the columns we need
	mat = dobj.limit_columns(headers)
	if not combine:
		minmaxes = []

		#find minima and maxima and stick them into two lists
		mins = mat.min(axis = 0)
		maxes = mat.max(axis = 0)

		for i in range(len(headers)):
			#put a pair of a min and max into minmaxes for every column
			minmaxes.append([mins[0,i],maxes[0,i]])
		return minmaxes
	else:
		return [mat.min(), mat.max()]
#the mean function returns a list of the means of each column

def mean(headers, dobj):
	return dobj.limit_columns(headers).mean(0)
#the stdev function returns a list of the stdevs of each column

def stdev(headers, dobj):
	return dobj.limit_columns(headers).std(0)
#normalize each data point based on local minima and maxima

def normalize_columns_separately(headers, dobj):
	minmaxes = data_range(headers, dobj)
	mat = dobj.limit_columns(headers)
	for i in range(mat.shape[1]):
		for j in range(mat.shape[0]):
			#translate the points so that the minimum is at 0
			mat[j,i] = mat[j,i] + minmaxes[i][0] * (-1)
			#scale the points so that they are all between 0 and 1
			mat[j,i] = mat[j,i] / (minmaxes[i][1] + minmaxes[i][0] * (-1))
	return mat
#normalize each data point based on the global minimum and maximum

def normalize_columns_together(headers, dobj):
	mat = dobj.limit_columns(headers)
	minimum = mat.min()
	maximum = mat.max()
	for i in range(mat.shape[1]):
		for j in range(mat.shape[0]):
			#translate the points so that the minimum is at 0
			mat[j,i] = mat[j,i] + (minimum * (-1))
			#scale the points so that they are all between 0 and 1
			mat[j,i] = mat[j,i] / (maximum + (minimum * (-1)))
	return mat
#Extension 1
#get the median of the entire matrix

def median(headers, dobj):
	#it needs to convert to an array or else it behaves strangely
	return np.median(np.asarray(dobj.limit_columns(headers)))
#Extension 2
#get the medians of each column individually


def median_separately(headers, dobj):
	medians = []
	for header in headers:
		#it needs to convert to an array or else it behaves strangely
		mat = np.asarray(dobj.limit_columns([header]))
		medians.append(np.median(mat))
	return medians

def main(argv):
	d = data.Data(filename=argv[1])
	headers = d.get_headers()
	print("headers",headers)
	res = linear_regression(d, [headers[0], headers[1]], headers[2])
	print('m0', res[0],
		  '\nm1', res[1],
		  '\nb', res[2],
		  '\nsse', res[3],
		  '\nr2', res[4],
		  '\nt', res[5],
		  '\np', res[6])

if __name__ == '__main__':
	main(sys.argv)