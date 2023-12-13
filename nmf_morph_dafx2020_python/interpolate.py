import numpy as np
from nmf_morph import NMFMorph
from untwist.untwist.data import Wave
from untwist.untwist.transforms import STFT, ISTFT
from rtpghi import pghi
from time import time
import librosa

print("Starting...")
start_time = time()

#src = "audio/Tremblay-BaB-HumDC-M.wav"
#tgt = "audio/Tremblay-SA-UprightPianoPedalWide.wav"
#src = "01.wav"
#tgt = "02.wav"
#src = "../01.wav"
#tgt = "../02.wav"

src = "0_120_cut.wav"
tgt = "1_125_cut.wav"
#src = "../0_120_cut.wav"
#tgt = "../1_125.wav"

fft_size = 2048
hop_size = 512
p = 0.9

print("Loading waves...")  # Wave and librosa produce same shape arrays (good)
proc_time = time()
s = Wave.read(src).to_mono()
t = Wave.read(tgt).to_mono()
slib, ssr = librosa.load(src, sr=None)
tlib, tsr = librosa.load(tgt, sr=None)
print(f"waveload done: {np.round((time()-proc_time)/60, decimals=2)}")

stft = STFT(fft_size=fft_size, hop_size=hop_size)
istft = ISTFT(fft_size=fft_size, hop_size=hop_size)

print("magnitude processing...")
proc_time = time()
S = stft.process(s).magnitude()
T = stft.process(t).magnitude()
#Slib = np.abs( librosa.stft(slib) )
#Tlib = np.abs( librosa.stft(tlib) )
print(f"magproc done: {np.round((time()-proc_time)/60, decimals=2)}")


# computation of frames we need to morph x seconds
assert ssr == tsr, "Sample rates of source and target are not equal."
morph_time = 3  # time in seconds
morph_meat = np.int16(np.ceil(morph_time / (hop_size / ssr)))  # final number of frames to morph
print(f"morph time: {morph_time}, frames: {morph_meat}")

print("NMFMorph (Wave)...")
proc_time = time()
m = NMFMorph()
S_morph = S[:, S.shape[1]-morph_meat:]  # selects last 3 seconds
T_morph = T[:, :morph_meat]  # selects first 3 seconds
m.analyze(S_morph, T_morph, p)
print(f"NMFMorph done: {np.round((time()-proc_time)/60, decimals=2)}")

"""
print("NMFMorph (librosa)...")
proc_time = time()
m = NMFMorph()
Slib_morph = Slib[:, S.shape[1]-morph_meat:]
Tlib_morph = Tlib[:, :morph_meat]
m.analyze(Slib, Tlib, p)
print(f"NMFMorph done: {np.round((time()-proc_time)/60, decimals=2)}")
"""

"""
for f in [0, 0.25, 0.5, 0.75, 0.95, 1]:
    print(f"f = {f}")
    Y = m.interpolate(f)
    y = istft.process(pghi(Y, fft_size, hop_size))
    y.write("out/%s%.2f.wav"%(name,f))
"""

print("Morphing...")
S_lastframe = S.shape[1]-morph_meat
s_lastsample = S_lastframe*hop_size
T_firstframe = morph_meat
t_firstsample = T_firstframe*hop_size

proc_time = time()
YY = S_morph  # init morph_spectrogram with its final shape: (fft_size/2+1, morph_meat) (the same as S_morph's and T_morph's)
YYY = s[:s_lastsample]
interpolation_factors = np.linspace(start=0, stop=1, num=morph_meat)  # goes in morph_meat steps from 0 to 1
for i, factor in enumerate(interpolation_factors):
    print(f"i: {i+1}/{morph_meat}, factor: {factor}")
    Y = m.interpolate(factor)
    YY[:, i:] = Y[:, i:]  # for a non-empty YY: overwrites this frame (each iteration overwrites the next frame)
    #YY.append(Y[:, i:])  # for an empty YY: appends this frame (each iteration overwrites the next frame) # no, this Spectrogram object doesn't know "append"
    
    Y = istft.process(pghi(Y, fft_size, hop_size))
    YYY = np.append(YYY, Y[i*512:(i+1)*512])
print(f"Morphing done: {np.round((time()-proc_time)/60, decimals=2)}")

"""
print("ISTFT on full file...")
proc_time = time()
yoog_array = np.append(S[:, :S.shape[1]-morph_meat], YY, axis=1)
yoog_array = np.append(yoog_array, T[:, morph_meat:], axis=1)
ready_file_YY = istft.process(pghi(yoog_array, fft_size, hop_size))
print(f"ISTFT done: {np.round((time()-proc_time)/60, decimals=2)}")
"""

print("Writing out full file...")
#ready_file_YY.write("morphed_whole_1.wav")

ready_file_YYY = np.append(YYY, t[t_firstsample:])
ready_file_YYY.write("morphed_whole_YYY1.wav")

"""
whole = s.append(YY)
whole = whole.append(t)
whole.write("morphed_whole2.wav")
"""

print(f"Done. Time taken: {np.round((time()-start_time)/60, decimals=2)}")
