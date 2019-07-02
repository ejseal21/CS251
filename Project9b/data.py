#Evan Arthur
#CS251
#2/22/19

import csv
import numpy as np
from datetime import datetime, timedelta
import analysis

class Data:

    def __init__(self, filename=None):
        self.kmeans = False
        if filename != None:
            self.read(filename)

    def read(self,filename):
        fp = open(filename, 'rU',encoding='utf-8-sig')
        csv_reader = csv.reader(fp)

        #headers: read in and stripped of whitespace
        self.headers = next(csv_reader)
        for head in self.headers:
            head = head.strip()

        #data types, reading in, then checking which ones
        #are non-numeric. Gets their indices, then will remove them.
        self.types = next(csv_reader)
        nonNumericCols = []
        dateCols = []
        count = 0
        for type in self.types:
            type = type.strip()
            if type == "date":
                dateCols.append(count)
            elif type != "numeric":
                nonNumericCols.append(count)
            count +=1

        self.data = []
        for line in csv_reader:
            row = []
            count = 0
            for item in line:
                item = item.strip()
                if count in dateCols:
                    #date = datetime.strptime(item,'%m/%d/%y')
                    date = self.parse_date(item)
                    row.append(float(self.convertTimeToNum(date)))
                    count += 1
                    continue
                if count in nonNumericCols:
                    count +=1
                    continue
                row.append(float(item))
                count += 1
            self.data.append(row)
        self.data = np.matrix(self.data)
        # print(self.data)

        #deletes the nonNumeric items of the headers and types
        for i in sorted(nonNumericCols, reverse=True):
            del self.headers[i]
            del self.types[i]
            # self.data = np.delete(self.data,i,1)

        self.header2col = {}
        count = 0
        for header in self.headers:
            self.header2col[header] = count
            count+=1

    def get_data(self):
        return self.data

    def get_headers(self):
        return self.headers

    def get_types(self):
        return self.types

    def get_num_dimensions(self):
        return self.data.shape[1]

    def get_num_points(self):
        return self.data.shape[0]

    def get_row(self,rowIndex):
        return self.data[(rowIndex-1):rowIndex]

    def get_value(self,header,rowIndex):
        colIndex = self.header2col[header]
        return self.data[rowIndex,colIndex]

    def getCol(self,header):
        return self.header2col[header]

    def set_kmeans(self,val=True):
        self.kmeans = val

    def get_kmeans(self):
        return self.kmeans

    def addCol(self,col):
        self.data = np.hstack((self.data,col))

    def writeFile(self,filename,headers=None):
        np.savetxt(filename)


    def newMatrix(self,colHeaders):
        cols = []
        for head in colHeaders:
            cols.append(self.header2col[head])
        new = self.data[:,cols]
        return new

    def parse_date(self,textDate):
        #tries a variety of different formats and throws an error if none work
        for format in ('%m-%d-%Y', '%m.%d.%Y', '%m/%d/%Y', '%m %d %Y', '%m-%d-%y', '%m.%d.%y', '%m/%d/%y', '%m %d %y'):
            try:
                return datetime.strptime(textDate, format)
            except ValueError:
                pass
        raise ValueError('no valid date format found')

    def convertTimeToNum(self, date):
        #takes in datetime object and returns a numeric representation
        epoch = datetime(1970, 1, 1)
        diff = date - epoch
        return (diff.days * 86400 + diff.seconds) * 10 ** 6 + diff.microseconds

    def convertNumToTime(self, date):
        #takes in a number and returns the string representation of the date
        epoch = datetime(1970, 1, 1)
        time = epoch + timedelta(microseconds=date)
        return time.strftime('%m/%d/%y')

    def __str__(self):
        #returns a string representation of the Data object
        string = ""
        string += str(self.headers) + "\n"
        string += str(self.types) + "\n"
        string += str(self.data) + "\n"
        return string

class PCAData(Data):
    #creates a PCAData object
    def __init__(self,pData,eVectors,eValues,ogmeans,ogheaders):
        Data.__init__(self)
        self.data = pData
        self.eVectors = eVectors
        self.eValues = eValues
        self.ogmeans = ogmeans
        self.ogheaders = ogheaders
        self.headers = ["PCA" + str(i) for i in range(len(ogheaders))]
        self.types = ["numeric" for i in range(len(ogheaders))]
        self.header2col = {}
        count = 0
        for header in self.headers:
            self.header2col[header] = count
            count+=1


    def get_eigenvalues(self):
        return self.eValues
    def get_eigenvectors(self):
        return self.eVectors
    def get_original_means(self):
        return self.ogmeans
    def get_original_headers(self):
        return self.ogheaders

class ClusterData(Data):
    def __init__(self,data,codebook,codes,errors,quality,K):
        Data.__init__(self)
        self.data = data.data
        # print("data: ", self.data)
        self.codebook = codebook
        print("codebook: ", self.codebook)
        self.codes = codes
        # print("codes: ", self.codes)
        self.errors = errors
        # print("errors: ", self.errors)
        self.quality = quality
        # print("quality: ", self.quality)
        self.K = K
        self.headers = data.get_headers()
        self.header2col = {}
        count = 0
        for header in self.headers:
            self.header2col[header] = count
            count+=1

    def getcodes(self):
        return self.codes

    def getCodebook(self):
        return self.codebook
    def getQuality(self):
        return self.quality



if __name__ == '__main__':
    data = Data(filename="testdata4.csv")
    print(data)
    bangBang = data.newMatrix(["datestuff","numberstuff"])
