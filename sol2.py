import numpy as np
import scipy.signal as sig
from scipy.misc import imread as imread
import matplotlib.pyplot as plt


def exp_formula(shape, u, arr, sign):
    '''
    compute the exp formula
    :param shape: the shape of the array
    :param u:
    :param sign: A boolean variable, where true means dft and false means idf
    :return: return a matrix that contains the calculation
    '''
    #compute the input to euler formula
    if sign:
        pre_input = np.divide(((-2j*np.pi)*u), shape)
    else:
        pre_input = np.divide(((2j * np.pi) * u), shape)
    exp_input =  arr *pre_input
    #return the calculation euler formula
    return np.exp(exp_input)

def DFT(signal):
    '''
    compute the discrete fourier transform
    :param signal: a vector that encapsulates a row in a matrix
    :return: An array of complex numbers that represent the signal in the frequency medium
    '''
    arr = np.arange(signal.shape[0])
    u = arr.reshape((signal.shape[0], 1))
    # compute the fft matrix using exp formula
    fft_matrix = exp_formula(signal.shape, u,arr , True)
    #multiply the fft matrix with the vector of the signal
    return np.dot(fft_matrix, signal)


def IDFT(fourier_signal):
    '''
    compute the inverse discrete fourier transform
    :param fourier_signal: a vector that encapsulates a row in a matrix
    :return: An array of complex numbers that represent the signal in the spatial medium
    '''
    arr = np.arange(fourier_signal.shape[0])
    u = arr.reshape((fourier_signal.shape[0], 1))
    # compute the fft matrix using euler formula

    fft_matrix = exp_formula(fourier_signal.shape, u, arr, False)
    # multiply the inverse fft matrix with the vector of the signal
    iFourier_array = np.dot(fft_matrix, fourier_signal)

    #divide by the length of the fourier signal
    normalized_idft = np.divide(iFourier_array,fourier_signal.shape)
    #todo decide if the origin array is real maybe use global variable
    #ignore the imaginary values
    #return np.real(normalized_idft)
    return normalized_idft

def dft_image(image ):
    fourier_image = np.asarray(image).astype(np.complex128)
    for row in range(image.shape[0]):
        fourier_image[row, :] = DFT(image[row, :])
    return fourier_image


def idft_image(fourier_image):
    img = np.asarray(fourier_image).astype(np.float64)
    for row in range(fourier_image.shape[0]):
        img[row, :] = IDFT(fourier_image[row, :])
    return img

def DFT2(image):
    arr = np.arange(image.shape[0])
    u = arr.reshape((image.shape[0], 1))
    fourier_image = dft_image(image)
    fft_matrix = exp_formula(image.shape[0], u, arr, True)
    return np.dot(fft_matrix, fourier_image)


def IDFT2(fourier_image):
    arr = np.arange(fourier_image.shape[0])
    u = arr.reshape((fourier_image.shape[0], 1))
    img = idft_image(fourier_image)

    fft_matrix = exp_formula(fourier_image.shape[0], u, arr, False)

    ifourier_matrix= np.divide(np.dot(fft_matrix, img),fourier_image.shape[0])
    return np.real(ifourier_matrix)



def main():
    name = "logoGray.jpg"
    img = imread(name)
    a= [[1.2, 2.4],[5.0, 7.1], [6.5, 3.4]]
    a=np.asarray(a).astype(np.float64)
    # fourier = np.fft.fft2(img)
    # ifourier = np.fft.ifft2(fourier).astype(np.uint8)
    # plt.imshow(ifourier)
    f= DFT2(a)
    i_f = IDFT2(f)
    v=4


if __name__ == "__main__":
    main()