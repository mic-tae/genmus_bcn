from untwist.untwist.transforms import STFT, ISTFT
from untwist.untwist.hpss import MedianFilterHPSS


use_librosa = False

# FFT and morph parameters
fft_size = 2048
hop_size = 512
p = 0.9
sr = None
morph_meat = None

# HPSS parameters
hLength = 17
pLength = 31

# Instantiate objects
stft = STFT(fft_size=fft_size, hop_size=hop_size)
istft = ISTFT(fft_size=fft_size, hop_size=hop_size)
hpss = MedianFilterHPSS(hLength, pLength)


## Files to use
#src = "../audio/01_cut.wav"
#tgt = "../audio/02_cut.wav"
#outfile = "../audio/0102cut"

#src = "../audio/0_120_cut.wav"
#tgt = "../audio/1_125_cut.wav"
#outfile = "../audio/120125cut"

#src = "../audio/00piano_cut.wav"
#tgt = "../audio/01trumpet_cut_resampled.wav"
#outfile = "../audio/0001_cut_test"

#src = "../audio/1_moretest125.wav"
#tgt = "../audio/2_moretest120.wav"
#outfile = "../audio/3_moretest_morphed_interpolate_normed"

#src = "../audio/finals/1_bpm120.wav"
#tgt = "../audio/finals/2_bpm115.wav"
#outfile = "../audio/finals/all_12_sigmoid"

#src = "../audio/finals/all_12_sigmoid_morphed_full.wav"
#tgt = "../audio/finals/3_bpm120.wav"
#outfile = "../audio/finals/all_123_sigmoid"

#src = "../audio/finals/all_123_sigmoid_morphed_full.wav"
#tgt = "../audio/finals/4_bpm125.wav"
#outfile = "../audio/finals/all_1234_sigmoid"

src = "../audio/finals/all_1234_sigmoid_morphed_full.wav"
tgt = "../audio/finals/5_bpm120.wav"
outfile = "../audio/finals/all_12345_sigmoid"
