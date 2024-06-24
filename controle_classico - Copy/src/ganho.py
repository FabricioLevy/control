import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import os
import pandas as pd
from functions import ControlFunctions

def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def generate_poles(real_start, real_end, real_step, imag_start, imag_end, imag_step):
    """ Generate complex conjugate pole pairs within specified ranges. """
    poles = []
    real_range = np.arange(real_start, real_end, real_step)
    imag_range = np.arange(imag_start, imag_end, imag_step)
    for real in real_range:
        for imag in imag_range:
            poles.append(complex(real, imag))
            poles.append(complex(real, -imag))
    return poles

def plot_root_locus(A, B, C, D, poles):
    """ Plot the root locus for given system matrices and a list of poles. """
    fig, ax = plt.subplots(figsize=(10, 6))
    for pole_set in [poles[i:i+6] for i in range(0, len(poles), 6)]:
        K = ctrl.place(A, B, pole_set)
        A_new = A - B.dot(K)
        sys = ctrl.ss(A_new, B, C, D)
        ctrl.root_locus(sys, Plot=True, ax=ax)
    ax.set_title('Root Locus for Varying Pole Configurations')
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')
    plt.show()

def main():
    fx = ControlFunctions()

    create_folder('../output/')
    create_folder('../input/')

    g = 9.8
    M_vazio = 29000
    L = 2.9
    C = 17
    H = 2
    rho_fluido = 715
    hf = 2 * 0.6
    mf = rho_fluido * L * C * hf

    m0 = mf - fx.massa_i(mf, 1, hf, L) - fx.massa_i(mf, 2, hf, L)
    m1 = fx.massa_i(mf, 1, hf, L)
    m2 = fx.massa_i(mf, 2, hf, L)
    m3 = fx.massa_i(mf, 3, hf, L)

    k1 = fx.calcular_ki(hf, L, g, mf, 1)
    k2 = fx.calcular_ki(hf, L, g, mf, 2)
    h0 = fx.calcular_hi(L, hf, 0)
    h1 = fx.calcular_hi(L, hf, 1)
    h2 = fx.calcular_hi(L, hf, 2)
    h3 = fx.calcular_hi(L, hf, 3)
    I0 = fx.momento_inercia(M_vazio, L, H)
    If = I0 + -m0 * h0**2 + m1 * h1**2 + m2 * h2**2

    omega1 = np.sqrt(k1 / m1)
    omega2 = np.sqrt(k2 / m2)
    zeta = ((6 * 10**(-6)) / (np.sqrt(g) * L**(3/2)))**(1/2)
    c1 = fx.calcular_ci(m1, omega1, zeta, 1)
    c2 = fx.calcular_ci(m2, omega2, zeta, 1)

    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [-k1 / m1, 0, g, -c1 / m1, 0, 0],
                  [0, -k2 / m2, g, 0, -c2 / m2, 0],
                  [g * m1 / If, g * m2 / If, -15000 * L**2 / (2 * If), 0, 0, -120 * L**2 / (2 * If)]])
    B = np.array([[0], [0], [0], [0], [0], [1 / If]])
    C = np.array([[0, 0, 1, 0, 0, 0]])
    D = np.array([[0]])

  
    real_start, real_end, real_step = -10, 0, 1
    imag_start, imag_end, imag_step = 0, 10, 1
    poles = generate_poles(real_start, real_end, real_step, imag_start, imag_end, imag_step)
    print(poles)
    quit()
    plot_root_locus(A, B, C, D, poles)

if __name__ == "__main__":
    main()
