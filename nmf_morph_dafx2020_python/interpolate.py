import numpy as np
from nmf_morph import NMFMorph
from untwist.untwist.data import Wave
from untwist.untwist.transforms import STFT, ISTFT
from rtpghi import pghi
from time import time

import librosa
import consts as cnst

# don't need this class anymore, but keep it for now
class Consts():
    def __init__(self):
        self.fft_size = 2048
        self.hop_size = 512
        self.p = 0.9
        self.stft = STFT(fft_size=self.fft_size, hop_size=self.hop_size)
        self.istst = ISTFT(fft_size=self.fft_size, hop_size=self.hop_size)
        self.sr = None

    def set_sr(self, sr):
        self.sr = sr


def calc_time(proc_t):
   return np.round((time()-proc_t)/60, decimals=2)


def load_waves(src, tgt):
    print("Loading waves...")  # Wave and librosa produce same shape arrays (good)
    proc_t = time()

    s = Wave.read(src).to_mono()
    t = Wave.read(tgt).to_mono()
    slib, ssr = librosa.load(src, sr=None)
    tlib, tsr = librosa.load(tgt, sr=None)

    assert ssr == tsr, "Sample rates of source and target are not equal."
    cnst.sr = ssr
    print(f"Loading waves done: {calc_time(proc_t)}")

    return s, t


def do_stft(s, t):
    print("magnitude processing...")
    proc_t = time()
    
    S = cnst.stft.process(s).magnitude()
    T = cnst.stft.process(t).magnitude()
    #Slib = np.abs( librosa.stft(slib) )
    #Tlib = np.abs( librosa.stft(tlib) )
    
    print(f"magproc done: {calc_time(proc_t)}")
    return S, T


def morph_parameter_calcs(morph_time, S, T):
    # computation of frames we need to morph x seconds
    morph_meat = np.int16( np.ceil(morph_time / (cnst.hop_size / cnst.sr)) )  # final number of frames to morph
    print(f"morph time: {morph_time}, frames: {morph_meat}")

    # select song parts for morphing
    Sm = S[:, S.shape[1]-morph_meat:]  # song 1: last [morph_meat] frames
    Tm = T[:, :morph_meat]  # song 2: first [morph_meat] frames

    return morph_meat, Sm, Tm


def do_nmf(Sm, Tm):
    print("NMFMorph (Wave)...")
    proc_t = time()

    m = NMFMorph()
    m.analyze(Sm, Tm, cnst.p)

    print(f"NMFMorph done: {calc_time(proc_t)}")

    """
    print("NMFMorph (librosa)...")
    proc_t = time()
    m = NMFMorph()
    Slib_morph = Slib[:, S.shape[1]-morph_meat:]
    Tlib_morph = Tlib[:, :morph_meat]
    m.analyze(Slib, Tlib, p)
    print(f"NMFMorph done: {np.round((time()-proc_t)/60, decimals=2)}")
    """

    return m, Sm, Tm


def do_morphing(S, Sm, s, m, morph_meat):
    """
    for f in [0, 0.25, 0.5, 0.75, 0.95, 1]:
        print(f"f = {f}")
        Y = m.interpolate(f)
        y = istft.process(pghi(Y, fft_size, hop_size))
        y.write("out/%s%.2f.wav"%(name,f))
    """
    print("Morphing...")
    
    S_lastframe = S.shape[1]-morph_meat
    s_lastsample = S_lastframe*cnst.hop_size
    T_firstframe = morph_meat
    t_firstsample = T_firstframe*cnst.hop_size

    YY = Sm  # "init" morph_spectrogram with its final shape: (fft_size/2+1, morph_meat) (the same as S_morph's and T_morph's)
    YYY = s[:s_lastsample]

    proc_t = time()
    interpolation_factors = np.linspace(start=0, stop=1, num=morph_meat)  # goes in morph_meat steps from 0 to 1
    for i, factor in enumerate(interpolation_factors):
        print(f"i: {i+1}/{morph_meat}, factor: {factor}")
        Y = m.interpolate(factor)
        YY[:, i:] = Y[:, i:]  # for a non-empty YY: overwrites this frame (each iter overwrites next frame)
        #YY.append(Y[:, i:])  # for an empty YY: appends this frame (each iter overwrites next frame)  # no! this Spectrogram object doesn't know "append"
        
        Y = cnst.istft.process(pghi(Y, cnst.fft_size, cnst.hop_size))  # super slow, same bad result
        YYY = np.append(YYY, Y[i*512:(i+1)*512])

    print(f"Morphing done: {calc_time(proc_t)}")
    return YY, YYY, t_firstsample


"""
print("ISTFT on full file...")
proc_t = time()
yoog_array = np.append(S[:, :S.shape[1]-morph_meat], YY, axis=1)
yoog_array = np.append(yoog_array, T[:, morph_meat:], axis=1)
ready_file_YY = istft.process(pghi(yoog_array, fft_size, hop_size))
print(f"ISTFT done: {np.round((time()-proc_t)/60, decimals=2)}")
"""


def write_file(YY, YYY, t_firstsample):
    print("Writing out full file...")
    # file based on appending frames
    #ready_file_YY.write("morphed_whole_1.wav")

    # file based on appending samples 
    ready_file_YYY = np.append(YYY, t[t_firstsample:])
    ready_file_YYY.write("morphed_whole_YYY1.wav")


if __name__ == "__main__":
    proc_t = time()

    s, t = load_waves(cnst.src, cnst.tgt)
    S, T = do_stft(s, t)
    morph_meat, Sm, Tm = morph_parameter_calcs(3, S, T)  # first parameter: time in seconds
    m, Sm, Tm = do_nmf(Sm, Tm)
    YY, YYY, t_firstsample = do_morphing(S, Sm, s, m, morph_meat)
    write_file(YY, YYY, t_firstsample)

    print(f"Done. Time taken: {calc_time(proc_t)}")
