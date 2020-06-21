import pywt
import numpy as np
import matplotlib.pyplot as plt

freq1 = 4
freq2 = 7
time = np.arange(0, 10, 0.01)

func1 = np.sin(2*np.pi*freq1*time)
func2 = np.sin(2*np.pi*freq2*time)

figure, axis = plt.subplots(4, 2)
plt.subplots_adjust(hspace=3)

axis[0, 0].set_title('Fungsi Sin Frequensi : 4 Hz')
axis[0, 0].plot(time, func1)
axis[0, 0].set_xlabel('Time')
axis[0, 0].set_ylabel('Amplitude')

axis[0, 1].set_title('Fungsi Sin Frequensi : 7 Hz')
axis[0, 1].plot(time, func2)
axis[0, 1].set_xlabel('Time')
axis[0, 1].set_ylabel('Amplitude')

func_merge = func1 + func2

axis[1, 0].set_title('Fungsi Sin dengan Frekuensi Gabungan')
axis[1, 0].plot(time, func_merge)
axis[1, 0].set_xlabel('Time')
axis[1, 0].set_ylabel('Amplitude')

fourierTransform = np.fft.fft(func_merge) / len(func_merge)
fourierTransform = fourierTransform[range(len(func_merge) // 2)]
tpCount = len(func_merge)
values = np.arange(tpCount // 2)
timePeriod = tpCount / 100
frequencies = values / timePeriod

axis[1, 1].set_title('Hasil dekonstruksi komponen frekuensi')
axis[1, 1].plot(frequencies, abs(fourierTransform))
axis[1, 1].set_xlabel('Frequency')
axis[1, 1].set_ylabel('Amplitude')


axis[2, 0].set_title('Fungsi Sin Frequensi : 4 Hz ')
axis[2, 0].plot(time, func1)
axis[2, 0].set_xlabel('Frequency')
axis[2, 0].set_ylabel('Amplitude')

[_, psi, xx] = pywt.Wavelet('sym4').wavefun(level=4)

axis[2, 1].set_title('Wavelet Symlets ')
axis[2, 1].plot(xx, psi)
axis[2, 1].set_xlabel('Frequency')
axis[2, 1].set_ylabel('Amplitude')

t = np.arange(0, 10, 0.01)
y = np.sin(2 * np.pi * 5 * t)
axis[3, 0].set_title('Fungsi Sin 5 Hz dan Wavelet ')
axis[3, 0].plot(xx, psi)
axis[3, 0].plot(t, y)
axis[3, 0].set_xlabel('Frequency')
axis[3, 0].set_ylabel('Amplitude')

plt.show()
