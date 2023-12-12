import numpy as np
from numpy.linalg import svd
from scipy.optimize import linear_sum_assignment
from untwist.untwist.data import Spectrogram
from untwist.untwist.factorizations import NMF
from optimal_transport import OptimalTransport


class NMFMorph:

    def get_rank(self, X, frac):
        u,s,vh = svd(X, full_matrices = False)
        total = np.sum(s)
        current = 0
        k = 0
        while (current / total) < frac:
            current += s[k]
            k += 1
        return k

    def analyze(self, S, T, p = 0.9, ks = None, kt = None):
        if ks == None: ks = self.get_rank(S, p)
        if kt == None: kt = self.get_rank(T, p)
        if kt < ks: kt = ks
        nmf_s = NMF(ks)
        nmf_t = NMF(kt)
        WS,HS, err = nmf_s.process(S)
        WT,HT, err = nmf_t.process(T)
        cost = np.zeros((ks, kt))
        ot_pairs = []
        for i in range(0, ks):
            ot_pairs_row = []
            for j in range (0, kt):
                ot = OptimalTransport(WS[:,i],WT[:,j])
                cost[i,j] = ot.distance
                ot_pairs_row.append(ot)
            ot_pairs.append(ot_pairs_row)
        rows,cols  = linear_sum_assignment(cost)
        self.ot = [ot_pairs[rows[i]][cols[i]] for i in range(ks)]
        self.rank = ks
        self.H = HS
        self.W = np.zeros_like(WS)
        self.src_rms = np.mean(S**2)

    def interpolate(self, factor):
        print(self.W.shape)
        for i in range(self.rank):
            self.W[:,i] = self.ot[i].interpolate(factor)
        V = np.dot(self.W, self.H)
        out_rms = np.sqrt(np.mean(V**2))
        g = self.src_rms / out_rms
        V = V * g
        return V
