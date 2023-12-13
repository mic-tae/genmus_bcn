import numpy as np
from nmf_morph import NMFMorph
from untwist.untwist.data import Wave
from rtpghi import pghi

from time import time
import librosa
import consts as cnst
import soundfile as sf



def calc_time(proc_t):
   return np.round((time()-proc_t)/60, decimals=2)


def load_waves(src, tgt):
    print(f"Loading waves: {src} and {tgt}...")  # Wave and librosa produce same shape arrays (good)
    proc_t = time()

    s = Wave.read(src).to_mono()
    t = Wave.read(tgt).to_mono()
    slib, ssr = librosa.load(src, sr=None)
    tlib, tsr = librosa.load(tgt, sr=None)

    assert ssr == tsr, "Sample rates of source and target are not equal."
    cnst.sr = ssr
    print(f"Waves loaded. {calc_time(proc_t)}")

    return s, t


def do_stft(s, t):
    print("Getting magnitudes...")
    proc_t = time()
    
    S = cnst.stft.process(s).magnitude()
    T = cnst.stft.process(t).magnitude()
    #Slib = np.abs( librosa.stft(slib) )
    #Tlib = np.abs( librosa.stft(tlib) )
    
    print(f"Magnitudes obtained. {calc_time(proc_t)}")
    return S, T


def morph_parameter_calcs(morph_time, S, T):
    # NOTE !! the beats/frames from beatnet are computed from 22kHz!!
    
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
    return m, Sm, Tm


def do_morphing(S, s, m, morph_meat):
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

    YY = np.empty((np.int16(cnst.fft_size/2+1), morph_meat))  # "init" morph_spectrogram with its final shape: (fft_size/2+1, morph_meat) (the same as S_morph's and T_morph's)
    YYY = s[:s_lastsample]

    proc_t = time()
    interpolation_factors = np.linspace(start=0, stop=1, num=morph_meat)  # goes in morph_meat steps from 0 to 1
    for i, factor in enumerate(interpolation_factors):
        print(f"i: {i+1}/{morph_meat}, factor: {factor}")
        Y = m.interpolate(factor)  # "Spectrogram" object, contains magnitudes
        Y_data = Y.as_ndarray()  # strip away the Spectrogram wrapper

        #YY[:, i:] = Y[:, i:]  # for a non-empty YY: overwrites this frame (each iter overwrites next frame)
        YY[:, i] = Y_data[:, i]  # for a non-empty YY: overwrites this frame (each iter overwrites next frame)
        #YY.append(Y[:, i:])  # for an empty YY: appends this frame (each iter overwrites next frame)  # no! this Spectrogram object doesn't know "append"
        
        #Y = cnst.istft.process(pghi(Y, cnst.fft_size, cnst.hop_size))  # super slow, same bad result
        #YYY = np.append(YYY, Y[i*512:(i+1)*512])

    print(f"Morphing done: {calc_time(proc_t)}")
    return YY, YYY, t_firstsample


def do_istft_modded_chunk(YY):
    print("ISTFT on modded chunk...")
    proc_t = time()
    
    ready_file_YY = librosa.istft(stft_matrix=YY, hop_length=cnst.hop_size, n_fft=cnst.fft_size)

    print(f"ISTFT done: {calc_time(proc_t)}")
    return ready_file_YY


def stitch_file(S, modded_chunk, T):
    """ 1. Takes the frames of the two input songs, frames were computed by the optimal transport library (from the Wave and Spectrogram objects)
        2. Cuts the areas of the  songs to where the morph part is
        3. Converts the frames to samples using librosa.istft """
    S = S[:, :S.shape[1]-morph_meat]  # song 1: until frame [morph_meat] begins
    T = T[:, morph_meat:]  # song 2: begins from frame [morph_meat]

    S = S.as_ndarray()
    T = T.as_ndarray()
    S_samples = librosa.istft(stft_matrix=S, hop_length=cnst.hop_size, n_fft=cnst.fft_size)
    T_samples = librosa.istft(stft_matrix=T, hop_length=cnst.hop_size, n_fft=cnst.fft_size)
    # at this point, Sm & modded_chunk & Tm are all samples
    
    ready_file = np.append(S_samples, modded_chunk)
    ready_file = np.append(ready_file, T_samples)

    return ready_file


def write_file_stitched(ready_file):
    print("Writing file...")
    sf.write("librosa_test__stitched_new.wav", ready_file, cnst.sr, "PCM_16")
    print("File written.")


def do_istft_fullfile(S, morph_meat, YY, T):
    print("ISTFT on full file...")
    proc_t = time()

    yoog_array = np.append(S[:, :S.shape[1]-morph_meat], YY, axis=1)
    yoog_array = np.append(yoog_array, T[:, morph_meat:], axis=1)
    ready_file_YY = cnst.istft.process(pghi(yoog_array, cnst.fft_size, cnst.hop_size))

    print(f"ISTFT done: {calc_time(proc_t)}")
    return ready_file_YY


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
    YY, YYY, t_firstsample = do_morphing(S, s, m, morph_meat)

    #ready_file_YY = do_istft_fullfile(S, morph_meat, YY, T)
    #write_file(YY, YYY, t_firstsample)

    ready_file_YY = do_istft_modded_chunk(YY)
    ready_file = stitch_file(S, ready_file_YY, T)
    write_file_stitched(ready_file)

    print(f"Done. Time taken: {calc_time(proc_t)}")
