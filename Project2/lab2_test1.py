# Bruce A. Maxwell
# CS 251 Project 2
# Spring 2018
# test file for data class
#

import data
import sys
import analysis

# test program for the Data class
def main(argv):

    # test command line arguments
    if len(argv) < 2:
        print( 'Usage: python %s <csv filename>' % (argv[0]))
        exit(0)

    # create a data object, which reads in the data
    dobj = data.Data(argv[1])

    # print out information about the data
    print('Number of rows:    ', dobj.get_num_points() )
    print('Number of columns: ', dobj.get_num_dimensions() )

    # print out the headers
    print("\nHeaders:")
    headers = dobj.get_headers()
    s = headers[0]
    for header in headers[1:]:
        s += ", " + header
    print( s )

    # print out the types
    print("\nTypes")
    types = dobj.get_types()
    s = types[0]
    for type in types[1:]:
        s += ", " + type
    print( s )

    # print out a single row
    print("\nPrinting row index 2")
    print( dobj.get_row( 2 ) )

    # print out all of the data
    print("\nData")
    headers = dobj.get_headers()
    print("headers:",headers)
    for i in range(dobj.get_num_points()):
            s = str( dobj.get_value( headers[0], i ) )
            for header in headers[1:]:
                s += "%10.3s" % (dobj.get_value( header, i ))
            print(s)

    print("\n\n\n\nselect_columns")

    d = dobj.get_data()
    # print("Data:", d)
    s = dobj.select_columns(['thing1', 'thing3'])
    print("Selected columns:", s)

    print("Data range:",analysis.data_range(['thing1', 'thing3'], dobj))
    print("Mean:",analysis.mean(['thing1', 'thing3'], dobj))
    print("Standard deviation:",analysis.stdev(['thing1', 'thing3'], dobj))
    print("Normalize columns separately:",analysis.normalize_columns_separately(['thing1', 'thing3'], dobj))
    print("Normalize columns together:",analysis.normalize_columns_together(['thing1', 'thing3'], dobj))

if __name__ == "__main__":
    main(sys.argv)