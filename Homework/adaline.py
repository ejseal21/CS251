'''
lecture36_adaline.py
ADALINE neural network trained on AND data
Oliver W. Layton
CS251: Data Analysis and Visualization
'''

import numpy as np


def adaline(data, classes, eta=0.1, nItr=10):
    '''Simulates an ADALINE network.
    Parameters:
        data: Input features (includes bias term). Assumed binary, either -1 or +1. MxN
        classes: Output classes. Assumed binary, -1  (c=0) or +1 (c=1). Mx1
        eta: Learning rate 0 < eta < 1
        nItr: Number of training passes
    '''
    # Number of features and data pts
    # Data: M training points x N features
    nFeatures = data.shape[1]
    nPoints = data.shape[0]

    # Set random seed for reproduceability
    np.random.seed(0)

    # Initialize weights randomly.
    w = 0.01*np.random.rand(1, nFeatures) - 0.01/2
    print('Initial weights:\n', w)

    for i in range(nItr):
        # Pass all training points through neural network before updating weights
        # data: MxN, w: 1xN -> Mx1
        y = data * w.T

        # Calculate error
        error = classes - y
        print('error after',i, ':',np.linalg.norm(error))
        y[y<0] = -1
        y[y>=0] = 1
        error = classes - y

        print('')
        # Loop through data training inputs, update weights
        for k in range(nPoints):
            w = w + eta*error[k, 0]*data[k, :]
        print(f'After iter {i}: weights:\n', w)
        print(f'After iter {i}: error:\n', error.T)
    return w

if __name__ == '__main__':
    #
    # AND input and classes
    # Format is [x1, x2, bias]
    and_input = np.matrix([[-1, -1, 1],
                          [1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]])
    and_classes = np.matrix([-1, -1, -1, 1]).T

    print(70*'-')
    print('AND input')
    print(70*'-')
    adaline(and_input, and_classes)

    xor_input = np.matrix([[-1, -1, 1],
                          [1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]])
    xor_classes = np.matrix([-1,1,1,-1])
    print(70*'-')
    print('XOR input')
    print(70*'-')
    adaline(xor_input, xor_classes)


    or_input = np.matrix([[-1, -1, 1],
                          [1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]])
    or_classes = np.matrix([-1,1,1,1])
    print(70*'-')
    print('OR input')
    print(70*'-')
    adaline(or_input, or_classes)