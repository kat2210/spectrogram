"""
Author: Katja Nell and Christine Ren, inspired by Mark Jay
Date: December 16, 2019 - March 29, 2020
Program: Spectrogram
"""

# preamble
import pyaudio, matplotlib.pyplot as plt, numpy as np, struct, sys
from matplotlib.ticker import EngFormatter, MultipleLocator, LogLocator
from scipy.fftpack import fft

# set constants
CHUNK = 1024 * 4  # samples per frame
FORMAT = pyaudio.paInt16  # audio format
CHANNELS = 1  # single channel for microphone
RATE = 44100  # samples per second

# create matplotlib figure and axes
fig, (ax, ax2) = plt.subplots(2, figsize=(10, 7))

p = pyaudio.PyAudio()
plt.ion()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=False,
    frames_per_buffer=CHUNK
)

# variables for plotting
x = np.arange(0, 2 * CHUNK, 2)  # samples for waveform
x_fft = np.linspace(0, RATE, CHUNK)  # frequencies for spectrum

# create a line object with random data
line, = ax.plot(x, np.zeros(CHUNK), 'b-', lw=1)

# create a semilog x line for spectrum
line_fft, = ax2.semilogx(x_fft, np.zeros(CHUNK), 'b-', lw=1, marker='.')

# axes formatting for waveform
ax.set_title('Audio Waveform')
ax.set_xlabel('Samples')
ax.set_ylabel('Volume')
ax.set_xlim(0, 2 * CHUNK)
ax.set_ylim(0, 255)
plt.setp(ax, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

# axis formatting for spectrum
ax2.set_title('Audio Spectrum')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Intensity')
ax2.set_xlim(20, RATE / 2)
ax2.set_ylim(0, 1.2)
formatter0 = EngFormatter(unit='Hz')
ax2.xaxis.set_major_formatter(formatter0)

# show plot
plt.tight_layout()
plt.show(block=False)


# close stuff
def handle_close(evt):
    print("Closing up...")
    stream.stop_stream()
    stream.close()
    sys.exit()


# updating value of main harmonic
def calc_primary_harmonic(y_fft, xdata):
    y_max = 0
    i = 0
    idx = 0
    np_abs = np.abs(y_fft[0:CHUNK])

    for y_mag in np_abs:
        if 20 <= xdata[i] < RATE / 2:  # Discard lower and upper harmonics as we cannot measure them - outside range of mic and contains noise
            if y_mag >= y_max:
                y_max = y_mag
                idx = i
        i += 1

    # http://www.add.ece.ufl.edu/4511/references/ImprovingFFTResoltuion.pdf
    if idx != len(np_abs) and idx > 0:
        y0 = np.max((np_abs[idx - 1], 1e-5))
        y1 = np.max((y_max,1e-5))
        y2 = np.max((np_abs[idx + 1], 1e-5))
        x1 = np.log(y2 / y0) / (2 * np.log(y1 ** 2 / (y0 * y2)))
        harmonic = RATE / CHUNK * (idx + x1)
    else:
        harmonic = xdata[idx]

    return np.max((0, harmonic))


fig.canvas.mpl_connect('close_event', handle_close)

# properties for textbox
props = dict(boxstyle='round', facecolor='midnightblue', alpha=0.1)

text = ax2.text(0.05, 0.95, "0", transform=ax2.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
while True:
    # binary data
    data = stream.read(CHUNK, False)

    # convert data to integers
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)

    # create np array and offset by 128
    data_np = np.array(data_int, dtype='b')[::2] + 128

    data_int = data_int * np.hanning(CHUNK * 2 + 1)[:-1]

    line.set_ydata(data_np)

    # compute FFT and update line
    y_fft = fft(data_int)
    # ax2.set_ylim(0, abs(y_fft[0]) /  (128 * CHUNK))   # use this to scale things real time
    y_fft[0] = 0 # Remove total harmonic
    line_fft.set_ydata(np.abs(y_fft[0:CHUNK]) / (128 * CHUNK))

    harmonic = calc_primary_harmonic(y_fft, line_fft.get_xdata())

    # set textbox data
    textstr = "Primary harmonic: {:.0f}".format(harmonic)
    # clear text each time to prevent overlaying buildup
    if text is not None:
        text.remove()
    # place textbox on graph
    text = ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

    # update figure canvas
    plt.pause(.0001)
