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
    return IDFT(IDFT(fourier_image).transpose()).transpose()


def conv_der(im):
    kernel = [[0,0,0],[1,0,-1],[0,0,0]]
    kernel = np.asarray(kernel)
    der_x = sig.convolve2d(im,kernel, mode='same')
    der_y = sig.convolve2d(im,kernel.transpose(), mode='same')
    return np.sqrt (np.abs(der_x)**2 + np.abs(der_y)**2)


def fourier_der(im):
    row = np.arange(im.shape[0]) -(im.shape[0]//2)
    row = np.asarray(row).reshape(len(row),1)
    col = np.arange(im.shape[1]) - (im.shape[1] // 2)


    frequency_signal = DFT2(im)
    shift_signal = np.fft.fftshift(frequency_signal)
    u_der = row* shift_signal
    v_der = col * shift_signal

    u_der = np.fft.ifftshift(u_der)
    v_der = np.fft.ifftshift(v_der)

    im_u = IDFT2(u_der)
    im_v  =IDFT2(v_der)
    return np.sqrt (np.abs(im_u)**2 + np.abs(im_v)**2)


def main():
    name = "gray_orig.png"
    img = imread(name)
    b= [[5.0, 7.1,5.6,7.6],[3.4,5.1,5.8,8.5],[1.3,3.5, 7.6,9.0]]
    b = np.asarray(b).astype(np.float64)
    magnitude = fourier_der(img)
    plt.imshow(magnitude,cmap= plt.cm.gray)
    plt.show()
    magnitude = conv_der(img)
    plt.imshow(magnitude, cmap=plt.cm.gray)
    plt.show()


if __name__ == "__main__":
    main()