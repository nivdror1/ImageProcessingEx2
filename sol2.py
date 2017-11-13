import numpy as np
import scipy.signal as sig
from scipy.misc import imread as imread
import matplotlib.pyplot as plt



def get_vandermonde_matrix(shape, u, arr, sign):
    '''
    compute the vandermonde matrix
    :param shape: the shape of the array
    :param u: (1,n) array containing the range (0, n-1)
    :param arr: (n,1) array containing the range (0,n-1)
    :param sign: A boolean variable, where true means dft and false means idf
    :return: return a matrix that contains the calculation
    '''

    if sign:
        return np.exp(-2j * np.pi * u * arr / shape)
    else:
        return np.exp(2j * np.pi * u * arr / shape)

def DFT(signal):
    '''
    compute the discrete fourier transform
    :param signal: a matrix of the signal in the spatial medium
    :return: An array of complex numbers that represent the signal in the frequency medium
    '''
    arr = np.arange(signal.shape[0])
    u = arr.reshape((signal.shape[0], 1))
    # compute the vandermonde matrix
    vandermonde_matrix = get_vandermonde_matrix(signal.shape[0], u, arr , True)
    #multiply the fft matrix with the vector of the signal
    return np.dot(vandermonde_matrix, signal)


def IDFT(fourier_signal):
    '''
    compute the inverse discrete fourier transform
    :param fourier_signal: a matrix of the signal in the frequency medium
    :return: An array of complex numbers that represent the signal in the spatial medium
    '''
    arr = np.arange(fourier_signal.shape[0])
    u = arr.reshape((fourier_signal.shape[0], 1))
    # compute the fft matrix using euler formula

    fft_matrix = get_vandermonde_matrix(fourier_signal.shape[0], u, arr, False)
    # multiply the inverse fft matrix with the vector of the signal
    iFourier_array = np.dot(fft_matrix, fourier_signal)

    #divide by the length of the fourier signal
    normalized_idft = np.divide(iFourier_array,fourier_signal.shape[0])
    #todo decide if the origin array is real maybe use global variable
    return normalized_idft


def DFT2(image):
    '''
    compute the discrete fourier transform for the 2d matrix
    :param image: A n*m matrix that contain values in the spatial medium
    :return: A complex n*m matrix that contain values in the frequency medium
    '''
    return DFT(DFT(image).transpose()).transpose()


def IDFT2(fourier_image):
    '''
    compute the inverse discrete fourier transform for the 2d matrix
    :param image: A n*m matrix that contain values in the frequency medium
    :return: A complex n*m matrix that contain values in the spatial medium
    '''
    # todo check i need to ignore the imaginary values
    return np.real(IDFT(IDFT(fourier_image).transpose()).transpose())


def conv_der(im):
    kernel = [[0,0,0],[1,0,-1],[0,0,0]]
    kernel = np.asarray(kernel)
    der_x = sig.convolve2d(im,kernel, mode='same')
    der_y = sig.convolve2d(im,kernel.transpose(), mode='same')
    return np.sqrt (np.abs(der_x)**2 + np.abs(der_y)**2)


def main():
    name = "monkeyGray.jpg"
    img = imread(name)
    b= [5.0, 7.1,5.6]
    b = np.asarray(b).astype(np.float64)
    magnitude = conv_der(img)
    plt.imshow(magnitude)


if __name__ == "__main__":
    main()