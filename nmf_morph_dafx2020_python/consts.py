from untwist.untwist.transforms import STFT, ISTFT


fft_size = 2048
hop_size = 512
p = 0.9

stft = STFT(fft_size=fft_size, hop_size=hop_size)
istst = ISTFT(fft_size=fft_size, hop_size=hop_size)

sr = None


#src = "01.wav"
#tgt = "02.wav"
#src = "../01.wav"
#tgt = "../02.wav"

src = "0_120_cut.wav"
tgt = "1_125_cut.wav"
#src = "../0_120_cut.wav"
#tgt = "../1_125_cut.wav"
