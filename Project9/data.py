import csv
import numpy as np
import sys
import analysis


class Data:
    def __init__(self, filename=None, headers=None, lists=None):
        self.filename = filename
        self.headers = headers
        self.types = []
        self.data = []
        self.header2col = {}
        self.file = None
        #if self.kmeans is true, that means that this was created using kmeans
        self.kmeans = False
        if self.filename is not None:
            self.read(self.filename)

    def write(self, filename, headers=None):
        if headers == None:
            tmp = self.headers
        else:
            tmp = self.limit_columns(headers)

        file = open(filename, 'w') 
        
        for i in range(len(tmp)):
            if tmp[i] == "None":
                del tmp[i]

        for i in range(len(tmp)):
            file.write(tmp[i])
            if i != len(tmp) - 1:
                file.write(', ')

        file.write("\n")

        for i in range(len(tmp)):
            file.write('numeric')
            if i != len(tmp) - 1:
                file.write(', ')
        
        file.write("\n")

        for i in range(len(self.data)):
            for j in range(len(tmp)):
                file.write(str(self.data[i,j]))
                if j != len(self.data)-1:
                    file.write(', ')
            file.write("\n")    
        print("file written to " + filename)

    def read(self, filename):
        # get our file
        file = open(filename, 'rU')
        # read it with the csv module
        csv_reader = csv.reader(file)
        # non_numeric will hold the indices of the non-numeric columns
        non_numeric = []
        x = 0
        # loop over the lines
        for line in csv_reader:
            if x == 0:  # the first line will be headers
                self.headers = line
                for i in range(len(self.headers)):
                    self.headers[i] = self.headers[i].strip()
            elif x == 1:  # the second line will be types
                self.types = line
                for i in range(len(self.types)):
                    self.types[i] = self.types[i].strip()
                    # add i to to the list of non-numeric indices so that
                    # we can remove all non-numeric headers, types, and data
                    if self.types[i] != 'numeric':
                        non_numeric.append(i)
            else:  # the rest will be data
                # self.data.append(line)
                for row in self.data:
                    for item in row:
                        item.strip()

                # temp_line allows self.data to be a 2d matrix
                temp_line = []
                for i in range(len(line)):
                    if self.types[i] == "numeric":
                        temp_line.append(line[i])
                self.data.append(temp_line)
            x += 1

        # This loop converts data to floats to do math on the data
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                self.data[i][j] = float(self.data[i][j])

        # reverse the array so that we don't get out of bounds errors
        non_numeric.reverse()
        for nn in non_numeric:
            del self.headers[nn]
            del self.types[nn]
        # i will be the index for header2col
        i = 0
        for header in self.headers:
            self.header2col[header] = i
            i += 1

        # make it a matrix
        self.data = np.matrix(self.data)
        return

 
    # limit the columns to whatever the user enters or to the first 10
    def limit_columns(self, headers=None):
        # initialize the list for the indices of columns we care about
        relevant_headers = []

        # check if the user entered an argument
        if headers == None:
            # if not, just take the first 10 columns
            if len(self.headers) > 2:
                relevant_headers = [i for i in range(2)]
            else:
                relevant_headers = [i for i in range(len(self.headers))]
        else:
            # if they did, get those indices
            for header in headers:
                if isinstance(header, int):
                    relevant_headers.append(header)
                else:
                    relevant_headers.append(self.header2col[header])
        return self.data[:, relevant_headers]

    # limit the rows to either the row indices the user enters or 0-9
    def limit_rows(self, indices=[i for i in range(10)]):
        return self.data[indices, :]

    #add a column using hstack
    def add_column(self, column):
        if column.shape[0] != self.data.shape[0] or column.shape[1] != 1:
            print("Something went wrong. You need a", self.data.shape[0], "by 1 numpy matrix.")
        else:
            #use hstack to put it on the right
            self.data = np.hstack((self.data, column))

    # add a row of data using vstack
    def add_point(self, values):
        # check that the length works
        if len(values) != len(self.headers):
            print("You need", len(self.headers), "values, but you gave", len(values))
        else:
            # use vstack to put it on the bottom
            self.data = np.vstack((self.data, values))

    # accessors
    def get_filename(self):
        return self.filename

    def get_headers(self):
        return self.headers

    def get_types(self):
        return self.types

    def get_data(self):
        return self.data

    def set_data(self, d):
        self.data = d

    def getHeader2col(self):
        return self.header2col

    def get_num_dimensions(self):
        return self.data[0].size

    def get_num_points(self):
        return len(self.data)

    def get_row(self, row_index):
        return self.data[row_index]

    def get_value(self, header, row_index):
        return self.data[row_index, self.header2col[header]]

    def set_kmeans(self, val=True):
        self.kmeans = val

    def get_kmeans(self):
        return self.kmeans



def main(argv):
    # test command line arguments
    if len(argv) < 2:
        print('Usage: python %s <csv filename>' % (argv[0]))
        exit(0)

    # create a data object, which reads in the data
    dobj = Data(argv[1])
    headers = dobj.get_headers()
    # test the five analysis functions
    print([headers[0], headers[2]])
    print("Data range by column:", analysis.data_range([headers[0], headers[2]], dobj))
    print("Mean:", analysis.mean([headers[0], headers[2]], dobj))
    print("Standard deviation:", analysis.stdev([headers[0], headers[2]], dobj))
    print("Normalize columns separately:", analysis.normalize_columns_separately([headers[0], headers[2]], dobj))
    print("Normalize columns together:", analysis.normalize_columns_together([headers[0], headers[2]], dobj))
    print("Median:", analysis.median([headers[0], headers[2]], dobj))
    print("Median Separately:", analysis.median_separately([headers[0], headers[2]], dobj))
    print("just  few rows:", dobj.limit_rows())
    print("just a few columns. I changed the limit to 2 for demonstration purposes:", dobj.limit_columns())
    print("Data range overall:", analysis.data_range([headers[0], headers[2]], dobj, True))
    print("The next two print statements get the last row of data. I add a row of data in between,"
          "so they are different.")
    print(dobj.get_row(-1))
    dobj.add_point([1, 2, 3])
    print(dobj.get_row(-1))

class PCAData(Data):
    def __init__(self, projected_data, eigenvectors=np.matrix([]), eigenvalues=np.matrix([]),
                 original_data_means=np.matrix([]), original_data_headers=[]):
    
        Data.__init__(self, headers=original_data_headers)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.mean_data_values = original_data_means
        self.original_data_headers = original_data_headers
        self.data = projected_data
        self.types = ['numeric' for _ in range(self.data[0].size)]
        self.headers = ['PCA' + str(i) for i in range(self.data[0].size)]
        self.header2col = {}
        for i in range(self.data[0].size):
            self.header2col[self.headers[i]] = i
        # print("self.header2col",self.header2col)

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_eigenvectors(self):
        return self.eigenvectors

    def get_original_means(self):
        return self.mean_data_values

    def get_original_headers(self):
        ret = []
        for header in self.original_data_headers:
            ret.append(header)
        return ret


if __name__ == "__main__":
    main(sys.argv)