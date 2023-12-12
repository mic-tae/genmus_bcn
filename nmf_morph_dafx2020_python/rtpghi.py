import numpy as np 
import heapq

'''
Phase recovery with 'real-time phase gradient heap integration'

https://ltfat.github.io/notes/ltfatnote043.pdf

TODO: Find faster heap implementaiton than heapq
'''

def _phase_deltas(logX,win_size,hop_size):
    gamma = 0.25645 * win_size**2 #assumes Hann window
    nbins = logX.shape[0]
    M = win_size
    a = hop_size
    m = np.arange(0,nbins)
    delta_f = 0.5 *  (np.roll(logX,-1,axis=1) - np.roll(logX,1,axis=1))
    delta_t = 0.5 *  (np.roll(logX,-1,axis=0) - np.roll(logX,1,axis=0))
    delta_f *= -gamma / (a*M)
    delta_t *=  a*M/gamma
    delta_t += 2 * np.pi * a * m[:,np.newaxis] / M  #this shift is in the paper, but I think I just remove it later...
    delta_t[0,:] = 0
    return delta_t,delta_f


def _pghi_frame(phase_freq_delta,phase_time_deltas,logmags,previous_phase,tolerence):
    '''
    see https://github.com/ltfat/ltfat/blob/master/libltfat/modules/libphaseret/src/rtpghi.c
    '''

    nbins = logmags.shape[0]

    flatlogs = logmags.reshape(2 * nbins,order='F')
    def heapcompare(i):
        return flatlogs[i]

    abstol = np.log(tolerence) + logmags[1:,:].max()
    delta_t = phase_time_deltas.reshape(2*nbins,order='F')
    phase_estimate = np.zeros(nbins)
    donemask = np.zeros((nbins))
    quickbreak = nbins
    heap = []

    # using the python heap queue, which is a min heap, not a max heap,
    # so negate on way in and way out
    for m in range(0,nbins):
        if logmags[m,1] <= abstol:
            donemask[m] = -1
            quickbreak -= 1
        else:
            heapq.heappush(heap,(-flatlogs[m],m))

    while quickbreak > 0 and heap:

        _,m = heapq.heappop(heap)

        if(m < nbins):
            m_next = m + nbins
            if(donemask[m] == 0):
                phase_estimate[m] = previous_phase[m] +\
                                        (delta_t[m] + delta_t[m_next]) * 0.5
                donemask[m] = 1
                heapq.heappush(heap,(-flatlogs[m_next],m_next))
                quickbreak -= 1
        else:
            m_prev = m - nbins
            if(m_prev != nbins - 1 and donemask[m_prev + 1] == 0):
                phase_estimate[m_prev + 1] = phase_estimate[m_prev] + \
                                            (phase_freq_delta[m_prev] + \
                                                phase_freq_delta[m_prev + 1]) * 0.5
                donemask[m_prev + 1] = 1
                heapq.heappush(heap,(-flatlogs[m + 1],m + 1))
                quickbreak -= 1
            if(m != 0 and donemask[m_prev - 1] == 0):
                phase_estimate[m_prev - 1] = phase_estimate[m_prev] -  \
                                            ( phase_freq_delta[m_prev] + \
                                                phase_freq_delta[m_prev - 1]) * 0.5
                donemask[m_prev - 1] = 1
                heapq.heappush(heap,(-flatlogs[m - 1],m - 1))
                quickbreak -= 1

    randidx = np.nonzero(donemask == -1)
    phase_estimate[randidx] = np.random.random((randidx[0].shape[0])) * np.pi * 2
    return phase_estimate

def pghi(X,win_size,hop_size,tolerence=1e-6):
    '''
    X: magnitudes
    tolerence: 0-1
    returns complex spectrogram
    '''
    logmags = np.log(X + np.spacing(1))
    (gradient_t,gradient_f) = _phase_deltas(logmags,win_size,hop_size)
    phase_estimate = np.zeros(X.shape)
    prevphases = np.zeros(X.shape)
    shift = np.linspace(0,1,X.shape[0]) * 2 * np.pi * hop_size

    for n in np.arange(0,X.shape[1]):

        idx = np.arange(n-1,n+1) % X.shape[1]
        nprev = n-1 % X.shape[1]
        prevphases[:,n] = phase_estimate[:,nprev]
        phase_estimate[:,n] = _pghi_frame(gradient_f[:,n],\
                                         gradient_t[:,idx],\
                                         logmags[:,idx],\
                                         phase_estimate[:,nprev],\
                                         tolerence)
    #we need to shift in time for this to work with signal.istft (different from matlab version)
    phase_estimate -= shift[:,np.newaxis]
    return X * np.exp(1j * phase_estimate)
