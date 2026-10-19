import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt


def ex1():
    target_iterations = 1e4
    time_step = 100

    b = base()
#    for i in range(target_iterations):
#        b.time_step()
#    b.time_step()

class base:
    def __init__(
        self,
        eF=0.75 * 1e-3 / 1e-9,
        sigma=10e-9,
        k=0.1e9,
        m=0.067 * const.m_e,
        L=200e-9,
        N=100,
        dt=1e-12,
    ):
        self.A0 = const.physical_constants["Bohr radius"][0]
        self.EH = const.physical_constants["Hartree energy"][0]  # J
        self.eF = eF * self.A0 / self.EH
        self.N = N
        self.L = L / self.A0  # in ab
        self.m = m / const.m_e
        self.dx = self.L / N  # in ab
        self.sigma = sigma / self.A0
        self.k = k * self.A0
        self.dt = dt / const.physical_constants["atomic unit of time"][0]
        self.__init_grid()

    def __init_grid(self):
        self.x = np.linspace(-self.L, self.L, 2 * self.N + 1)

        self.psi = np.zeros((2, 2 * self.N + 1), dtype=complex)
        self.psi[0][1 : 2 * self.N] = np.exp(
            -self.x[1 : 2 * self.N] ** 2 / (2 * self.sigma**2)
            + 1j * self.k * self.x[1 : 2 * self.N]
        )
        self.normalize(axis=0)
        self.psi[1][1 : 2 * self.N] = self.psi[0][1 : 2 * self.N] * np.exp(
            -1j * self.Eavg(self.psi[0])
        )

    def normalize(self, axis=1):
        C = np.sqrt(self.dx * (self.psi[axis].conj() * self.psi[axis]).sum())
        self.psi[axis] /= C

    def V(self, x):
        return self.eF * x

    def Hamiltonian(self, psi):
        """
        Hamiltonian returns H * \\Psi
        """
        return (
            -(
                psi[2 : 2 * self.N + 1]
                + psi[0 : 2 * self.N - 1]
                - 2 * psi[1 : 2 * self.N]
            )
            / (2 * self.m * self.dx**2)
            + self.V(self.x[1 : 2 * self.N]) * psi[1 : 2 * self.N]
        )

    def time_step(self):
        self.psi[0][1 : 2 * self.N], self.psi[1][1 : 2 * self.N] = (
            self.psi[1][1 : 2 * self.N],
            self.psi[0][1 : 2 * self.N]
            + 2 * self.dt / 1j * self.Hamiltonian(self.psi[1]),
        )

    def Eavg(self, psi):
        return self.dx * np.sum(psi[1 : 2 * self.N].conj() * self.Hamiltonian(psi))
