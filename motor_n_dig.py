# Guardamos el contenido del motor N basado en POO en un archivo descargable para José María

import numpy as np

class MotorN:
    def __init__(self, dim=50, gamma=0.05, lambda_=0.02):
        self.dim = dim
        self.gamma = gamma
        self.lambda_ = lambda_
        self.rho = np.random.rand(dim, dim) * 0.1
        self.tiempo = 0

    def laplaciano(self, campo):
        return (
            -4 * campo
            + np.roll(campo, 1, axis=0)
            + np.roll(campo, -1, axis=0)
            + np.roll(campo, 1, axis=1)
            + np.roll(campo, -1, axis=1)
        )

    def entropia_local(self, campo):
        kernel = np.ones((3, 3)) / 9
        from scipy.signal import convolve2d
        mean = convolve2d(campo, kernel, mode="same", boundary="wrap")
        varianza = convolve2d((campo - mean) ** 2, kernel, mode="same", boundary="wrap")
        return np.sqrt(varianza)

    def evolucionar(self):
        delta_rho = (
            self.gamma * self.laplaciano(self.rho)
            - self.lambda_ * self.entropia_local(self.rho)
        )
        self.rho += delta_rho
        self.rho = np.clip(self.rho, 0, 1)
        self.tiempo += 1

    def inyectar(self, x, y, intensidad=1.0):
        if 0 <= x < self.dim and 0 <= y < self.dim:
            self.rho[y, x] += intensidad
            self.rho = np.clip(self.rho, 0, 1)

    def reiniciar(self):
        self.rho = np.random.rand(self.dim, self.dim) * 0.1
        self.tiempo = 0

    def obtener_rho(self):
        return self.rho

    def obtener_entropia_global(self):
        s_local = self.entropia_local(self.rho)
        return np.mean(s_local)

    def inyectar(self, x, y, intensidad=1.0):
        if 0 <= x < self.dim and 0 <= y < self.dim:
            self.rho[y, x] += intensidad
            self.rho = np.clip(self.rho, 0, 1)

    def reiniciar(self):
        self.rho = np.random.rand(self.dim, self.dim) * 0.1
        self.tiempo = 0

