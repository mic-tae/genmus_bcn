from untwist.untwist.transforms import STFT, ISTFT


use_librosa = False

fft_size = 2048
hop_size = 512
p = 0.9

stft = STFT(fft_size=fft_size, hop_size=hop_size)
istft = ISTFT(fft_size=fft_size, hop_size=hop_size)

sr = None
morph_meat = None


#src = "01.wav"
#tgt = "02.wav"
#src = "../01_cut.wav"
#tgt = "../02_cut.wav"

#src = "0_120_cut.wav"
#tgt = "1_125_cut.wav"
#src = "../0_120_cut.wav"
#tgt = "../1_125_cut.wav"


#src = "00piano_cut.wav"
#tgt = "01trumpet_cut.wav"
src = "../00piano_cut.wav"
tgt = "../01trumpet_cut_resampled.wav"
