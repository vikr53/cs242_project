import numpy as np

class Quantize:
    def init():
        self.bitwidth = 8
   
    # Uniform quantization
    def quantize(self, W):
        n = self.bitwidth
        if n >= 32:
            return W
        range = np.abs(np.min(W)) + np.abs(np.max(W))
        d = range / (2**(n))
        z = -np.min(W, 0) // d
        W = np.rint(W / d)
        W = d * (W)
        return W