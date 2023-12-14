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


def resample_song(filename, sr):
    cnst.sr = sr
    tlib, tsr = librosa.load(filename, sr=cnst.sr)
    write_file(tlib, f"{filename.replace('.wav', '_resampled.wav')}")


def load_waves(src, tgt):
    print(f"Loading waves: {src} and {tgt}...")  # Wave and librosa produce same shape arrays (good)
    proc_t = time()

    s = Wave.read(src).to_mono()
    t = Wave.read(tgt).to_mono()
    slib, ssr = librosa.load(src, sr=None)
    tlib, tsr = librosa.load(tgt, sr=None)

    # currently unused block (using assert below instead)
    #if ssr != tsr:
    #    print(f"WARNING: Sample rates: source {ssr} != target {tsr}. Setting target = source.")
    #    tlib, tsr = librosa.load(tgt, sr=ssr)
    assert ssr == tsr, f"WARNING: Sample rates: source {ssr} != target {tsr}. Export the song again first."
    
    cnst.sr = ssr
    print(f"Waves loaded. {calc_time(proc_t)}")

    return s, t


def get_max_amplitudes(s, t):
    cnst.amp_s_max = np.max(s)
    cnst.amp_t_max = np.max(t)


def do_stft(s, t):
    print("Getting magnitudes...")
    proc_t = time()
    
    if cnst.use_librosa:
        S = np.abs( librosa.stft(s) )
        T = np.abs( librosa.stft(t) )
    else:
        S = cnst.stft.process(s).magnitude()
        T = cnst.stft.process(t).magnitude()
    
    print(f"Magnitudes obtained. {calc_time(proc_t)}")
    return S, T


def calc_morph_parameters(morph_time, S, T):
    # NOTE !! the beats/frames from beatnet are computed from 22kHz!!
    
    cnst.morph_meat = np.int16( np.ceil(morph_time / (cnst.hop_size / cnst.sr)) )  # computes num of frames we need to morph [morph_tine] seconds

    # select song chunks for morphing
    Sm = S[:, S.shape[1]-cnst.morph_meat:]  # song 1: last [morph_meat] frames
    Tm = T[:, :cnst.morph_meat]  # song 2: first [morph_meat] frames

    print(f"Morph parameters -- time: {morph_time} seconds, frames: {cnst.morph_meat}")
    return Sm, Tm


def do_nmf(Sm, Tm):
    print("NMFMorph (Wave)...")
    proc_t = time()

    m = NMFMorph()
    m.analyze(Sm, Tm, cnst.p)

    print(f"NMFMorph done: {calc_time(proc_t)}")
    return m


def do_morphing(S, m):
    print("Morphing...")
    
    # init morphed_chunk 
    if cnst.use_librosa:
        morphed_chunk = np.empty((np.int16(cnst.fft_size/2+1), cnst.morph_meat))  # "init" with final shape (fft_size/2+1, morph_meat) (the same as S_morph's and T_morph's)
    else:
        morphed_chunk = S[:, :cnst.morph_meat].copy()  # copy important! "init" Spectrogram object with final shape (fft_size/2+1, morph_meat)
    print(f"morphed_chunk initialized, shape: {morphed_chunk.shape}")

    proc_t = time()
    morph_factors = np.linspace(start=0, stop=1, num=cnst.morph_meat+2)  # +2 because we are going to remove [0] and [-1] for they are 0 and 1.
    morph_factors = np.delete(np.delete(morph_factors, 0), -1)
    for i, factor in enumerate(morph_factors):
        print(f"i: {i+1}/{cnst.morph_meat}, factor: {factor}")
        Y = m.interpolate(factor)  # "Spectrogram" object with mags

        if cnst.use_librosa:
            Y_data = Y.as_ndarray()  # strips the Spectrogram wrapper
            morphed_chunk[:, i] = Y_data[:, i]  # overwrites this frame (each iter overwrites next frame)
        else:
            morphed_chunk[:, i] = Y[:, i]

    print(f"Morphing done: {calc_time(proc_t)}")
    return morphed_chunk


def do_istft_morphed_chunk(morphed_chunk):
    """ takes a spectrogram object / spectrogram matrix, converts it to samples, returns samples """
    print("ISTFT on morphed chunk...")
    proc_t = time()

    if cnst.use_librosa:
        morphed_chunk = morphed_chunk.as_ndarray()
        morphed_chunk_samples = librosa.istft(stft_matrix=morphed_chunk, hop_length=cnst.hop_size, n_fft=cnst.fft_size)
    else:
        morphed_chunk_samples = cnst.istft.process(pghi(morphed_chunk, cnst.fft_size, cnst.hop_size))
        morphed_chunk_samples = morphed_chunk_samples.normalize()

    print(f"ISTFT done: {calc_time(proc_t)}")
    return morphed_chunk_samples


def stitch_file(S, morphed_chunk_samples, T):
    """ This is mainly for testing purposes -- to see the morphed chunk in its original context.

        1. Takes the frames of the two input songs
        2. Cuts the areas of the songs to where the morph part is
        3. Converts the frames to samples
        4. Stitches the chopped sample blocks back together
    
        ### Note: morphed_chunk_samples is already SAMPLES and doesn't need to be ISTFT'd ###
    """
    print("Stitching blocks...")
    proc_t = time()

    S = S[:, :S.shape[1]-cnst.morph_meat]  # song 1: until frame [morph_meat] begins
    T = T[:, cnst.morph_meat:]  # song 2: begins from frame [morph_meat]

    if cnst.use_librosa:
        S = S.as_ndarray()
        T = T.as_ndarray()
        S_samples = librosa.istft(stft_matrix=S, hop_length=cnst.hop_size, n_fft=cnst.fft_size)
        T_samples = librosa.istft(stft_matrix=T, hop_length=cnst.hop_size, n_fft=cnst.fft_size)
        ready_file = np.append(S_samples, morphed_chunk_samples)
        ready_file = np.append(ready_file, T_samples)
    else:
        S_samples = cnst.istft.process(pghi(S, cnst.fft_size, cnst.hop_size))  # shape (xxx, 1)
        T_samples = cnst.istft.process(pghi(T, cnst.fft_size, cnst.hop_size))
        ready_file = np.append(S_samples, morphed_chunk_samples, axis=0)
        ready_file = np.append(ready_file, T_samples, axis=0)

    print(f"Stitching done: {calc_time(proc_t)}")
    return ready_file


def write_file(samples, filename):
    print(f"Writing {filename}...")
    sf.write(filename, samples, cnst.sr, "PCM_16")
    print("File written.")


def main():
    proc_t = time()

    s, t = load_waves(cnst.src, cnst.tgt)
    get_max_amplitudes(s, t)
    S, T = do_stft(s, t)
    Sm, Tm = calc_morph_parameters(3, S, T)  # first parameter: time in seconds
    m = do_nmf(Sm, Tm)
    morphed_chunk = do_morphing(S, m)
    morphed_chunk_samples = do_istft_morphed_chunk(morphed_chunk)
    write_file(morphed_chunk_samples, f"{cnst.outfile}_morphed_chunk.wav")

    ## this is mainly for testing purposes - to see the modded chunk in its original context
    ready_file = stitch_file(S, morphed_chunk_samples, T)
    write_file(ready_file, f"{cnst.outfile}_morphed_full.wav")

    print(f"All done: {calc_time(proc_t)}")


if __name__ == "__main__":
    main()
    #resample_song(cnst.tgt)
