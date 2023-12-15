from nmf_morph import NMFMorph
from untwist.untwist.data import Wave
from rtpghi import pghi
import consts as cnst

from interpolate import *

from songs_concatenation import Song_concatenator, Beat_detector, BEATNET_PATH

from time import time
import librosa
import argparse



def stitch_file_time(s, morphed_chunk_samples, t):
    """ This is mainly for testing purposes -- to see the morphed chunk in its original context.

        1. Takes the frames of the two input songs
        2. Cuts the areas of the songs to where the morph part is
        3. Converts the frames to samples
        4. Stitches the chopped sample blocks back together
    
        ### Note: morphed_chunk_samples is already SAMPLES and doesn't need to be ISTFT'd ###
    """
    print("Stitching blocks...")
    proc_t = time()
    
    # if cnst.use_librosa:
    #    S = S.as_ndarray()
    #    T = T.as_ndarray()
    #    S_samples = librosa.istft(stft_matrix=S, hop_length=cnst.hop_size, n_fft=cnst.fft_size)
    #    T_samples = librosa.istft(stft_matrix=T, hop_length=cnst.hop_size, n_fft=cnst.fft_size)
    #    ready_file = np.append(S_samples, morphed_chunk_samples)
    #    ready_file = np.append(ready_file, T_samples)
    #else:
    #    S_samples = cnst.istft.process(pghi(S, cnst.fft_size, cnst.hop_size))  # shape (xxx, 1)
    #    T_samples = cnst.istft.process(pghi(T, cnst.fft_size, cnst.hop_size))
    #    ready_file = np.append(S_samples, morphed_chunk_samples, axis=0)
    #    ready_file = np.append(ready_file, T_samples, axis=0)

    morphed_chunk_samples = morphed_chunk_samples.as_ndarray()
    ready_file = np.concatenate([s, morphed_chunk_samples])
    ready_file = np.concatenate([ready_file, t])
    print(f"Stitching done: {calc_time(proc_t)}")
    return ready_file



def morphing_two_songs(source, target, outfile, concatenator, sample_rate=44100):
    proc_t = time()

    print("Loading songs...")
    audio1, sr = librosa.load(source)
    audio2, _ = librosa.load(target)
    s, t = load_waves(source, target)
    get_max_amplitudes(s, t)
    
    # Calculate morph_time
    print("Detecting downbeats and calculating morphing time...")
    concantenated_song, last_beat, first_beat = concatenator.concatenate(audio1, audio2, sr)
    s_duration = len(s)/sample_rate
    morph_time = s_duration - last_beat[0]
    
    # NOTE Just trying to make the morph time a bit more accurate
    #if morph_time > first_beat[0]:
    #    extra_samples = int((morph_time - first_beat[0])*sample_rate)
    #    s = s[:-int(extra_samples)]
    #    morph_time = first_beat[0]
    #    first_cut = int(last_beat[0]*cnst.sr)-extra_samples
    #    second_cut = int(first_beat[0]*cnst.sr)
    #else:
    #    extra_samples = int((morph_time - first_beat[0])*sample_rate)
    #    t = t[int(extra_samples):]
    #    first_beat[0] = morph_time
    #    first_cut = int(last_beat[0]*cnst.sr)
    #    second_cut = int(first_beat[0]*cnst.sr)
    
    # STFT
    S, T = do_stft(s, t)
    
    # Morphing
    Sm, Tm = calc_morph_parameters(morph_time, S, T)  # first parameter: time in seconds
    m = do_nmf(Sm, Tm)
    morphed_chunk = do_morphing(S, m)
    morphed_chunk_samples = do_istft_morphed_chunk(morphed_chunk)
    write_file(morphed_chunk_samples, f"{cnst.outfile}_morphed_chunk.wav")

    ## this is mainly for testing purposes - to see the modded chunk in its original context
    first_cut = int(last_beat[0]*cnst.sr)
    second_cut = int(first_beat[0]*cnst.sr)
    s = s.as_ndarray()
    t = t.as_ndarray()
    s = s[:first_cut]  # song 1: until sample [last_beat_sample] begins
    t = t[second_cut:]  # song 2: begins from sample [first_beat_sample] until the end

    ready_file = stitch_file_time(s, morphed_chunk_samples, t)
    write_file(ready_file, f"{cnst.outfile}_morphed_full.wav")
    sf.write("../audio/test/test_concatenated_song.wav", concantenated_song, sr, "PCM_16")
    
    print(f"All done: {calc_time(proc_t)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPM calculator script")
    parser.add_argument('--first_song', type=str, help='Path to the first song', default=cnst.src)
    parser.add_argument('--second_song', type=str, help='Path to the second song', default=cnst.tgt)
    parser.add_argument('--save_path', type=str, help='Path where to save concatenated song', default="test")
    args = parser.parse_args()
    
    beatnet = Beat_detector(1, BEATNET_PATH, "offline", 'DBN', sample_rate=22050)
    song_concatenator = Song_concatenator(beatnet, 3)
    
    print(f"Morphing two songs...")
    morphing_two_songs(source=args.first_song, target=args.second_song, outfile=args.save_path, concatenator=song_concatenator, sample_rate=44100)
    print("Songs morphed and concatenated successfully!")
