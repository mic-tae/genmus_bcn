from nmf_morph import NMFMorph
from untwist.untwist.data import Wave
from rtpghi import pghi
import consts as cnst

from interpolate import *

from songs_concatenation import Song_concatenator, Beat_detector, BEATNET_PATH

from time import time
import librosa
import argparse

def morphing_two_songs(source, target, outfile, concatenator, sample_rate=44100):
    proc_t = time()

    audio1, sr = librosa.load(source)
    audio2, _ = librosa.load(target)
    s, t = load_waves(source, target)
    get_max_amplitudes(s, t)
    S, T = do_stft(s, t)
    
    # Calculate morph_time
    print("Calculating morph time...")
    concantenated_song, last_beat, first_beat = concatenator.concatenate(audio1, audio2, sr)
    
    s_duration = len(s)/sample_rate
    morph_time = s_duration - last_beat[0]
    
    Sm, Tm = calc_morph_parameters(morph_time, S, T)  # first parameter: time in seconds
    m = do_nmf(Sm, Tm)
    morphed_chunk = do_morphing(S, m)
    morphed_chunk_samples = do_istft_morphed_chunk(morphed_chunk)
    write_file(morphed_chunk_samples, f"{cnst.outfile}_morphed_chunk.wav")

    ## this is mainly for testing purposes - to see the modded chunk in its original context
    ready_file = stitch_file(S, morphed_chunk_samples, T)
    write_file(ready_file, f"{cnst.outfile}_morphed_full.wav")
    write_file(concantenated_song, "concatenated_song.wav")

    print(f"All done: {calc_time(proc_t)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPM calculator script")
    parser.add_argument('--first_song', type=str, help='Path to the first song', default="../audio/test/house120.mp3")
    parser.add_argument('--second_song', type=str, help='Path to the second song', default="../audio/test/2house120.mp3")
    parser.add_argument('--save_path', type=str, help='Path where to save concatenated song', default="test")
    args = parser.parse_args()
    
    beatnet = Beat_detector(1, BEATNET_PATH, "offline", 'DBN', sample_rate=22050)
    song_concatenator = Song_concatenator(beatnet, 3)
    
    print("Morphing...")
    morphing_two_songs(source=args.first_song, target=args.second_song, outfile=args.save_path, concatenator=song_concatenator, sample_rate=44100)
    print("Song concatenated successfully!")

