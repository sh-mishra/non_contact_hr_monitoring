import numpy
from scipy import signal
from matplotlib import pyplot as plt
from math import sqrt, log, exp, pi

def autocorr(x):
    result = numpy.correlate(x, x, mode='full')
    return result[int(result.size/2):]

if __name__ == '__main__':

    dat = numpy.arange(0,100)
    arr1 = (2*numpy.pi/4)*dat
    arr2 = (2*numpy.pi/5)*dat
    #arr3 = (2*numpy.pi/2)*dat
    s1 = numpy.cos(arr1)
    s2 = numpy.sin(arr2)
    #s3 = numpy.cos(arr3)
    sig = s1+s2

    cor = autocorr(sig)

    plt.figure(1)
    
    plt.subplot(2,1,1)
    plt.plot(sig)
    plt.ylabel("Data")
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(cor)
    plt.ylabel("Cepstrum")
    plt.grid(True)
    
    plt.show()
