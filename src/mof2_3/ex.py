import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import time


def ex3(F=0.75 * 1e-3 / 1e-9, out="ex3", target_iterations=1e4):
    print("Ex3: calculating psi evolution")

    start = time.perf_counter()

    target_iterations = int(target_iterations)
    time_step = 100

    b = base(eF=F)
    p = []
    for i in range(target_iterations):
        if i % time_step == 0:
            p.append(np.abs(b.psi[0]) ** 2)

        b.time_step()

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        b.x * b.A0,
        np.linspace(0, target_iterations, int(target_iterations / time_step)),
        p,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("$\\Psi$")
    ax.set_xlabel("Położenie [m]")
    ax.set_ylabel("Czas [dt]")
    fig.savefig(f"{out}.svg", format="svg")
    plt.close(fig)

    end = time.perf_counter()
    print(f"Ex3: Done. Took {end-start:.4f} seconds")


def ex4(out="ex4", F=0.75 * 1e-3 / 1e-9, target_iterations=1e4):
    print("Ex4: calculating expected position evolution")

    start = time.perf_counter()

    target_iterations = int(target_iterations)
    time_step = 100
    b = base(eF=F)
    xs = []
    xs_analitical = []
    for i in range(target_iterations):
        if i % time_step == 0:
            xs.append(b.xExp())
            xs_analitical.append(b.k * i / b.m - b.eF * i**2 / (2 * b.m))

        b.time_step()

    t = np.linspace(0, target_iterations, int(target_iterations / time_step))
    fig, ax = plt.subplots()

    ax.plot(t, np.array(xs) * b.A0, label="Rozwiązanie numeryczne")
    ax.plot(t, np.array(xs_analitical) * b.A0, label="Rozwiązanie analityczne")
    ax.legend()
    ax.set_xlabel("Czas [dt]")
    ax.set_ylabel("$\\left<x\\right>$ [m]")
    ax.grid()

    fig.savefig(f"{out}.svg", format="svg")
    plt.close(fig)

    end = time.perf_counter()
    print(f"Ex3: Done. Took {end-start:.4f} seconds")


def ex5():
    print("Ex5: recalling Ex3 and Ex4 with no external field")
    ex3(out="ex5_mesh", F=0)
    ex4(out="ex5_expected_x", F=0)


class base:
    def __init__(
        self,
        eF=0.75 * 1e-3 / 1e-9,
        sigma=10e-9,
        k=0.1e9,
        m=0.067 * const.m_e,
        L=200e-9,
        N=100,
        dt=1,
    ):
        self.A0 = const.physical_constants["Bohr radius"][0]
        self.EH = const.physical_constants["Hartree energy"][0]  # J
        self.eF = eF * const.e * self.A0 / self.EH
        self.N = N
        self.L = L / self.A0  # in ab
        self.m = m / const.m_e
        self.dx = self.L / N  # in ab
        self.sigma = sigma / self.A0
        self.k = k * self.A0
        self.dt = dt  # lets say dt is already in AU. # / const.physical_constants["atomic unit of time"][0]
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
            -1j * self.Eavg(self.psi[0]) * self.dt
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

    def xExp(self, ax=0):
        return self.dx * np.sum(np.abs(self.psi[ax]) ** 2 * self.x)
