import numpy as np

class ACP:
    def __init__(self, X):
        self.X = X
        self.Cov = np.cov(m=X, rowvar=False)
        self.valoriProp, self.vectoriProp = np.linalg.eigh(a=self.Cov)
        k_desc = [k for k in reversed(np.argsort(self.valoriProp))]
        print(k_desc)
        self.alpha = self.valoriProp[k_desc]
        self.A = self.vectoriProp[:, k_desc]
        self.C = self.X @ self.A
        self.Rxc = self.A * np.sqrt(self.alpha)

    def getValoriProp(self):
        return self.alpha

    def getVectoriProp(self):
        return self.A

    def getCompPrin(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc


