# Bruce A. Maxwell
# CS 251 Project 5
# Test code for single linear regression

import sys
import data
import analysis

# Reads in a file and executes a single linear regression using the specified columns
def main(argv):

    if len(argv) < 4:
        print("Usage: python %s <data file> <independent header> <dependent header>")
        exit(-1)

    # read some data
    data_obj = data.Data( argv[1] )
    ind_header = argv[2]
    dep_header = argv[3]

    # call the analysis function
    results = analysis.single_linear_regression( data_obj, ind_header, dep_header)

    # print out the results
    print("Model:    y = %.4fx + %.4f" % (results[0], results[1]) )
    print("R-value:  %.3f" % (results[2]) )
    print("P-value:  %.3f" % (results[3]) )
    print("Stderr:   %.3f" % (results[4]) )

    return


if __name__ == "__main__":
    main(sys.argv)