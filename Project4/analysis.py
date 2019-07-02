import numpy as np
import data

#Extension 5
#return minimum value-maximum value for each column
def data_range(headers, dobj, combine = False):
	#make a matrix of just the columns we need
	mat = dobj.limit_columns(headers)
	if combine == False:
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
	return (dobj.limit_columns(headers).std(0))

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
