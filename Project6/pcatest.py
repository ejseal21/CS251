# Bruce A. Maxwell
# Spring 2018
# Updated for python3
# CS 251 Project 6
#
# PCA test function
#
import numpy as np
import data
import analysis
import sys
# import pcadata

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python %s <data file>' % (sys.argv[0]))
        exit()

    data = data.Data( sys.argv[1] )
    pcadata = analysis.pca( data, data.get_headers(), False )

    print("\nOriginal Data Headers")
    print(pcadata.get_original_headers())
    print("\nOriginal Data" )
    print(data.limit_columns( data.get_headers() ))
    print("\nOriginal Data Means")
    print(pcadata.get_original_means())
    print("\nEigenvalues")
    print(pcadata.get_eigenvalues())
    print("\nEigenvectors")
    print(pcadata.get_eigenvectors())
    print("\nProjected Data")
    print(pcadata.limit_columns(pcadata.get_headers()))
