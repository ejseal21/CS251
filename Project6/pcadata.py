import csv
import numpy as np
import sys
import analysis
import data

class PCAData(data.Data):
    def __init__(self, projected_data, eigenvectors=np.matrix([]), eigenvalues=np.matrix([]),
             
                 original_data_means=np.matrix([]), original_data_headers=[]):
        data.Data.__init__(headers=original_data_headers)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.mean_data_values = original_data_means
        self.original_data_headers = original_data_headers
        self.data = projected_data
        self.types = ['numeric' for _ in range(self.data[0].size)]
        self.headers = ['PCA' + str(i) for i in range(self.data[0].size)]
        print(self.headers)
        self.header2col = {}
        for i in range(self.data[0].size):
            self.header2col[self.headers[i]] = i

    def get_eigenvalues(self):
        return self.eigenvalues #np.matrix([[eigenvalue for eigenvalue in self.eigenvalues]])

    def get_eigenvectors(self):
        return self.eigenvectors

    def get_original_means(self):
        return self.mean_data_values

    def get_original_headers(self):
        return self.original_data_headers.copy()
    