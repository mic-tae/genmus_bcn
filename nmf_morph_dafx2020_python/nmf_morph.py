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

    def scaling(self, V):
        out_rms = np.sqrt(np.mean(V**2))
        g = self.src_rms / out_rms
        V = V * g
        return V

    def interpolate(self, factor):
        #print(self.W.shape)
        for i in range(self.rank):
            self.W[:,i] = self.ot[i].interpolate(factor)
        V = np.dot(self.W, self.H)
        return self.scaling(V)

    def smooth_fade(self, factors):
        """
        m (ex: (1025, 259)) consists of frames from V, with V = W x H
        W is W[:, i] = ot[i].interpolate(factor)
        -> first idea: for each W[:, i], we could interpolate with another factor
        that won't work because W (1025, 30), and H (30, 259) -> we need a frame from H!
        so we need a full W to compute W * H and obtain (1025, 259)
        -> second idea: only compute one (the current) frame from H using full W
        instead of computing all of H and then just picking one frame of it
        BUT: the scaling will be (slightly) different
        because previously H was computed fully
        """
        assert len(factors) == self.H.shape[1], f"error: len(factors) {len(factors)} not equal to H.shape[1] {self.H.shape[1]}"        
        
        V = np.zeros((self.W.shape[0], self.H.shape[1]))
        for frame in range(self.H.shape[1]):
            print(f"Morphing frame {frame+1}/{self.H.shape[1]}...")
            for i in range(self.rank):
                self.W[:, i] = self.ot[i].interpolate(factors[frame])  # always need to compute W fully
            V[:, frame] = np.dot(self.W, self.H[:, frame])  # but can compute only one frame of H
        return self.scaling(V)  # need to check in the end of the scaling makes things horrible
