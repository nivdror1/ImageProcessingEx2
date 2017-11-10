import numpy as np
import scipy.signal as sig


def euler_formula(signal, u, sign):
    '''
    compute euler formula
    :param signal: a vector that encapsulates a row in a matrix
    :param u:
    :param sign: A boolean variable, where true means dft and false means idf
    :return: return a matrix that contains the calculation
    '''
    #compute the input to euler formula
    if sign:
        pre_input = np.divide(((-2*np.pi)*u), signal.shape)
    else:
        pre_input = np.divide(((2 * np.pi) * u), signal.shape)
    euler_input =  signal *pre_input
    #return the calculation euler formula
    return np.cos(euler_input) + (np.sin(euler_input)*1j)


def DFT(signal):
    '''
    compute the discrete fourier transform
    :param signal: a vector that encapsulates a row in a matrix
    :return: An array of complex numbers that represent the signal in the frequency medium
    '''
    arr = np.arange(signal.shape[0])
    u = arr.reshape((signal.shape[0], 1))
    # compute the fft matrix using euler formula
    fft_matrix = euler_formula(signal, u, True)
    #multiply the fft matrix with the vector of the signal
    return np.dot(fft_matrix, signal)


def Idft(fourier_signal):
    '''
    compute the inverse discrete fourier transform
    :param fourier_signal: a vector that encapsulates a row in a matrix
    :return: An array of complex numbers that represent the signal in the spatial medium
    '''
    arr = np.arange(fourier_signal.shape[0])
    u = arr.reshape((fourier_signal.shape[0], 1))
    # compute the fft matrix using euler formula
    fft_matrix = euler_formula(fourier_signal, u, False)
    # multiply the inverse fft matrix with the vector of the signal
    iFourier_array = np.dot(fft_matrix, fourier_signal)
    #divide by the length of the fourier signal
    return np.divide(iFourier_array,fourier_signal.shape)


def main():
    a= [3.5,5.6, 4.6]
    signal = np.array(a).astype(np.float64)
    fourier= DFT(signal)
    check_fourier = np.fft.fft(a)
    f=4

if __name__ == "__main__":
    main()