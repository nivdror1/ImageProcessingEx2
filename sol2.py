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
    '''
    compute the magnitude of the image by using convolution with the ReLU kernel
    :param im: The input image as a matrix
    :return: return the magnitude
    '''
    # The ReLU kernel
    kernel = [[0,0,0],[1,0,-1],[0,0,0]]
    kernel = np.asarray(kernel)

    # The x and y derivative using convolution with the ReLU kernel
    der_x = sig.convolve2d(im,kernel, mode='same')
    der_y = sig.convolve2d(im,kernel.transpose(), mode='same')
    # compute the magnitude
    return np.sqrt (np.abs(der_x)**2 + np.abs(der_y)**2)


def fourier_der(im):
    '''
    Compute the magnitude of the image by using the DFT
    :param im: The input image as a matrix
    :return: return the magnitude
    '''
    # Set up the array indices
    row = np.arange(im.shape[0]) -(im.shape[0]//2)
    row = np.asarray(row).reshape(len(row), 1)
    col = np.arange(im.shape[1]) - (im.shape[1] // 2)

    # Perform the DFT
    frequency_signal = DFT2(im)
    # Shift the frequency signal so the (0,0) pixel will be in the center
    shift_signal = np.fft.fftshift(frequency_signal)

    # Compute the x and y derivative
    u_der = row* shift_signal
    v_der = col * shift_signal

    # Shift the frequency signal so the (0,0) pixel will be in the top-left
    u_der = np.fft.ifftshift(u_der)
    v_der = np.fft.ifftshift(v_der)

    # Perform the IDFT
    im_u = IDFT2(u_der)
    im_v  =IDFT2(v_der)

    # Compute the magnitude
    return np.sqrt (np.abs(im_u)**2 + np.abs(im_v)**2)


def get_gaussian_kernel(kernel_size):
    '''
    Compute the gaussian kernel
    :param kernel_size: The length of the row/column of the kernel
    :return: The matrix that contain the gaussian kernel
    '''
    kernel_row = [1, 1]
    bin_vec = kernel_row.copy()
    #if the size is one return the gaussian kernel [1]
    if kernel_size == 1:
        gaussian_kernel = [1]
        return np.asarray(gaussian_kernel)

    # use convolution to achieve the binomy co-efficient
    while kernel_size != 2:
        kernel_row = sig.convolve(kernel_row, bin_vec)
        kernel_size -=1
    # get the matrix and divide it the by the sum of the values
    kernel_col = kernel_row.reshape(len(kernel_row), 1)
    gaussian_kernel =  kernel_row*kernel_col
    return gaussian_kernel/gaussian_kernel.sum()

def blur_spatial(im, kernel_size):
    # todo deal with 1 kernel size
    # todo deal with im as float 64
    # todo deal with the padding
    gaussian_kernel = get_gaussian_kernel(kernel_size)
    return sig.convolve2d(im,gaussian_kernel, mode='same').astype(np.uint8)

def main():
    name = "monkeyGray.jpg"
    img = imread(name)
    b= [[5.0, 7.1,5.6,7.6],[3.4,5.1,5.8,8.5],[1.3,3.5, 7.6,9.0]]
    b = np.asarray(b).astype(np.float64)
    # magnitude = fourier_der(img)
    # plt.imshow(magnitude,cmap= plt.cm.gray)
    # plt.show()
    # magnitude = conv_der(img)
    # plt.imshow(magnitude, cmap=plt.cm.gray)
    # plt.show()
    rim = blur_spatial(img, 1)
    plt.imshow(rim, cmap=plt.cm.gray)
    plt.show()



if __name__ == "__main__":
    main()