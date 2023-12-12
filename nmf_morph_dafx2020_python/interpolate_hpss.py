from nmf_morph import NMFMorph
from untwist.data import Wave
from untwist.transforms import STFT, ISTFT
from untwist.hpss import MedianFilterHPSS
from rtpghi import pghi

src = "audio/Tremblay-BaB-HumDC-M.wav"
tgt = "audio/Tremblay-SA-UprightPianoPedalWide.wav"
name = "humpiano_hpss"


p = 0.8
fft_size = 2048
hop_size = 512

hLength = 17
pLength = 31


s = Wave.read(src).to_mono()
t = Wave.read(tgt).to_mono()

stft = STFT(fft_size=fft_size, hop_size = hop_size)
istft = ISTFT(fft_size=fft_size, hop_size = hop_size)
hpss = MedianFilterHPSS(hLength, pLength)

S = stft.process(s)
T = stft.process(t)

Hs, Ps = hpss.process(S)
Ht, Pt = hpss.process(T)

m_h = NMFMorph()
m_p = NMFMorph()
m_h.analyze(Hs.magnitude(), Ht.magnitude(), p)
m_p.analyze(Ps.magnitude(), Pt.magnitude(), p)

for f in [0, 0.25, 0.5, 0.75, 0.95, 1]:
    Yh = m_h.interpolate(f)
    Yp = m_p.interpolate(f)
    yh = istft.process(pghi(Yh, fft_size, hop_size))
    yp = istft.process(pghi(Yp, fft_size, hop_size))
    y = yh + yp
    y.write("out/%s%.2f.wav"%(name,f))
