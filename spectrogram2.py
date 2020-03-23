import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import struct
import pyaudio
from scipy.fftpack import fft

import sys
import time


class AudioStream(object):
    def __init__(self):

        # pyqtgraph stuff
        pg.setConfigOptions(antialias=True)
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        dim = self.app.desktop().screenGeometry()
        self.win = pg.GraphicsWindow(title='Spectrum Analyzer')
        self.win.setWindowTitle('Spectrum Analyzer')
        self.win.setGeometry(5, 5, dim.width()-2*5, dim.height()-5)

        wf_xlabels = [(0, '0'), (2048, '2048'), (4096, '4096')]
        wf_xaxis = pg.AxisItem(orientation='bottom')
        wf_xaxis.setTicks([wf_xlabels])

        wf_ylabels = [(0, '0'), (127, '128'), (255, '255')]
        wf_yaxis = pg.AxisItem(orientation='left')
        wf_yaxis.setTicks([wf_ylabels])

        sp_xlabels = [
            (np.log10(10), '10'), (np.log10(100), '100'),
            (np.log10(1000), '1000'), (np.log10(10000), '10000')
        ]
        sp_xaxis = pg.AxisItem(orientation='bottom')
        sp_xaxis.setTicks([sp_xlabels])

        self.waveform = self.win.addPlot(
            title='<b><font size="4" color=white>Audio Waveform</font></b>', row=1, col=1, axisItems={'bottom': wf_xaxis, 'left': wf_yaxis},
        )
        self.spectrum = self.win.addPlot(
            title='<b><font size="4" color=white>Audio Spectrum</font></b>', row=2, col=1, axisItems={'bottom': sp_xaxis},
        )

        # pyaudio stuff
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 2

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=False,
            frames_per_buffer=self.CHUNK,
        )
        # waveform and spectrum x points
        self.x = np.arange(0, 2 * self.CHUNK, 2)
        self.f = np.linspace(0, self.RATE, self.CHUNK)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='c', width=3)
                self.waveform.setYRange(0, 255, padding=0)
                self.waveform.setXRange(0, 2 * self.CHUNK, padding=0.005)
                self.waveform.setLabel('left', text='<font color=white>Volume</font>')
                self.waveform.setLabel('bottom', text='<font color=white>Samples</font>')
            if name == 'spectrum':
                self.traces[name] = self.spectrum.plot(pen='g', width=3)
                self.spectrum.setLogMode(x=True, y=False)
                self.spectrum.setYRange(0, 1.2)
                self.spectrum.setXRange(
                    np.log10(20), np.log10(self.RATE / 2), padding=0.005)
                self.spectrum.setLabel('left', text='<font color=white>Intensity</font>')
                self.spectrum.setLabel('bottom', text='<font color=white>Frequency (Hz)</font>')


    def update(self):
        wf_data1 = self.stream.read(self.CHUNK, False)
        wf_data2 = struct.unpack(str(2 * self.CHUNK) + 'B', wf_data1)
        wf_data3 = np.array(wf_data2, dtype='b')[::2] + 128
        self.set_plotdata(name='waveform', data_x=self.x, data_y=wf_data3)

        sp_data = fft(wf_data2)
        sp_data2 = np.abs(sp_data[0:self.CHUNK]) / (128 * self.CHUNK)
        self.set_plotdata(name='spectrum', data_x=self.f, data_y=sp_data2)
        self.harmonic = float(self.RATE) / float(len(sp_data2)) * self.calc_primary_harmonic(sp_data2)

    def update2(self):
        self.spectrum.setTitle('<b><font size="4" color=white>Audio Spectrum</font></b> - <i><font size="3" color=white>' + "{0:.2f}".format(self.harmonic) + "Hz" '</font></i>')

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.animation2()
        self.start()

    def animation2(self):
        self.timer2 = QtCore.QTimer()
        self.timer2.timeout.connect(self.update2)
        self.timer2.start(200)

    #updating value of main harmonic
    def calc_primary_harmonic(self, y_fft):
        y_max = 0
        i = 0
        idx = 0
        for y_mag in y_fft:
            if self.f[i] >= 1 and self.f[i] < self.RATE / 2:
                if y_mag >= y_max:
                    y_max = y_mag
                    idx = i
            i += 1
        return idx

if __name__ == '__main__':

    audio_app = AudioStream()
    audio_app.animation()
