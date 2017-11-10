import numpy as np
import scipy.signal as sig


def euler_formula(signal, u):
    '''
    compute euler formula
    :param signal: a vector that encapsulates a row in a matrix
    :param u:
    :return: return a matrix that contains the calculation
    '''
    #compute the input to euler formula
    pre_input = np.divide(((-2*np.pi)*u), signal.shape)
    euler_input = pre_input * signal
    #return the calculation euler formula
    return np.cos(euler_input) + (np.sin(euler_input)*1j)


def DFT(signal):
    '''
    compute the discrete fourier transform
    :param signal: a vector that encapsulates a row in a matrix
    :return:
    '''
    arr = np.arange(signal.shape[0])
    u = arr.reshape((signal.shape[0], 1))
    euler_matrix = euler_formula(signal, u)
    return np.dot(euler_matrix, signal)


def main():
    a= [3.5,5.6, 4.6]
    signal = np.array(a).astype(np.float64)
    fourier= DFT(signal)
    d=3

if __name__ == "__main__":
    main()