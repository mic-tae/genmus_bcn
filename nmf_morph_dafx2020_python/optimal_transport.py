import numpy as np

class OptimalTransport:
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.distance = 0
        self.a_masses = self.segment(A)
        self.b_masses = self.segment(B)
        self.matrix = self.compute_matrix(self.a_masses, self.b_masses)


    def detect_peaks(self, x):
      dx = x[1:] - x[:-1]
      peaks = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
      return peaks

    def segment(self, X):
      total_mass = np.sum(X) + np.spacing(1)
      valleys = self.detect_peaks(-X)
      if(valleys[0]!=0):
          valleys = np.concatenate(([0], valleys))
      masses = []
      for i in range(len(valleys)-1):
        start = valleys[i]
        end = valleys[i+1]
        center = start + np.argmax(X[start:end])
        mass = np.sum(X[start:end]) / total_mass
        masses.append([valleys[i],valleys[i+1],center,mass])
      return masses

    def compute_matrix(self, m1, m2):
        i1 = 0
        i2 = 0
        mass1 = m1[0][3]
        mass2 = m2[0][3]
        matrix = []
        while(True):
            if mass1 < mass2:
                matrix.append([i1, i2,mass1])
                self.distance += mass1 * (i1-i2)**2
                mass2 = mass2 - mass1
                i1= i1 + 1
                if i1 >= len(m1):
                    break
                mass1 = m1[i1][3]
            else:
                matrix.append([i1, i2, mass2])
                self.distance += mass2 * (i1-i2)**2
                mass1 = mass1 - mass2
                i2 = i2 + 1
                if i2 >= len(m2):
                    break
                mass2 = m2[i2][3]
        return matrix

    def interp_mass(self, pos, m):
        dif = pos - m[2]
        start = max(0, m[0] + dif)
        end = min(len(self.A), m[1] + dif)
        return (start, end)

    def interpolate(self, alpha):
        result = np.zeros_like(self.A)
        for pair in self.matrix:
            m1 = self.a_masses[pair[0]]
            m2 = self.b_masses[pair[1]]
            interp = int(np.round((1 - alpha)* m1[2] + alpha * m2[2]))
            fr1, to1 = self.interp_mass(interp, m1)
            s1 = (1 - alpha) * pair[2] / m1[3]
            seg1 = s1 * self.A[m1[0]:m1[1]]
            if(to1-fr1) != len(seg1): continue
            result[fr1:to1] += seg1
            fr2, to2 = self.interp_mass(interp, m2)
            s2 = alpha * pair[2] / m2[3]
            seg2 = s2 * self.B[m2[0]:m2[1]]
            if(to2-fr2) != len(seg2): continue
            result[fr2:to2] += seg2
        return result
