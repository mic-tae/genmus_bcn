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

#src = "0_120_cut.wav"
#tgt = "1_125.wav"
src = "../0_120_cut.wav"
tgt = "../1_125.wav"
name = "humpiano"

fft_size = 2048
hop_size = 512
p = 0.9

print("Loading waves...")
proc_time = time()
s = Wave.read(src).to_mono()  # Wave and librosa produce same shape arrays (good)
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
#Slib = librosa.stft(slib)
#Tlib = librosa.stft(tlib)
#Slib = np.abs(Slib)
#Tlib = np.abs(Tlib)
print(f"magproc done: {np.round((time()-proc_time)/60, decimals=2)}")


# computation of frames we need to morph x seconds
assert ssr == tsr, "NOOOO THE SAMPLE RATES ARE NOT THE SAME!!!!!!!"
morph_time = 3  # time in seconds
morph_meat = np.int16(np.ceil(morph_time / (hop_size / ssr)))  # final number of frames to morph


print("NMFMorph (Wave)")
proc_time = time()
m = NMFMorph()
S_morph = S[:, S.shape[1]-morph_meat:]
T_morph = T[:, :morph_meat]
m.analyze(S_morph, T_morph, p)
print(f"NMFMorph done: {np.round((time()-proc_time)/60, decimals=2)}")

"""
print("NMFMorph (librosa)")
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
proc_time = time()
Y = None
YY = S_morph
interpolation_factors = np.linspace(start=0, stop=1, num=morph_meat)  # goes in morph_meat steps from 0 to 1
for i, factor in enumerate(interpolation_factors):
    print(f"i: {i}/{morph_meat}, current morph factor: {factor}")
    Y = m.interpolate(factor)
    YY[:, i:] = Y[:, i:]  # only appends the necessary frame (per iteration, start from one frame further)
print(f"NMFMorph done: {np.round((time()-proc_time)/60, decimals=2)}")
YY = istft.process(pghi(YY, fft_size, hop_size))
YY.write("morphed2.wav")

print("Writing out full file...")
whole = s.append(YY)
whole = whole.append(t)
whole.write("morphed_whole2.wav")

print(f"Done. Time taken: {np.round((time()-start_time)/60, decimals=2)}")
