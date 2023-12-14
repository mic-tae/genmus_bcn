from untwist.untwist.transforms import STFT, ISTFT


use_librosa = False

fft_size = 2048
hop_size = 512
p = 0.9

stft = STFT(fft_size=fft_size, hop_size=hop_size)
istft = ISTFT(fft_size=fft_size, hop_size=hop_size)

sr = None
morph_meat = None


#src = "../audio/01_cut.wav"
#tgt = "../audio/02_cut.wav"
#outfile = "../audio/0102cut"

src = "../audio/0_120_cut.wav"
tgt = "../audio/1_125_cut.wav"
outfile = "../audio/120125cut"

#src = "../audio/00piano_cut.wav"
#tgt = "../audio/01trumpet_cut_resampled.wav"
#outfile = "../audio/0001_cut"
