import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import os
import pandas as pd
from functions import ControlFunctions

def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

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

    polos_list = [
        # [-1 - 30j, -1 + 30j, -2 - 28j, -2 + 28j, -3 - 26j, -3 + 26j],
        # [-2 - 28j, -2 + 28j, -3 - 26j, -3 + 26j, -4 - 24j, -4 + 24j],
        [-3 - 26j, -3 + 26j, -4 - 24j, -4 + 24j, -5 - 22j, -5 + 22j],
        # [-3 - 28j, -3 + 28j, -4 - 26j, -4 + 26j, -5 - 24j, -5 + 24j],
        # [-4 - 24j, -4 + 24j, -5 - 22j, -5 + 22j, -6 - 20j, -6 + 20j],
        # [-5 - 22j, -5 + 22j, -6 - 20j, -6 + 20j, -7 - 18j, -7 + 18j],
        # [-6 - 20j, -6 + 20j, -7 - 18j, -7 + 18j, -8 - 16j, -8 + 16j],
        # [-7 - 18j, -7 + 18j, -8 - 16j, -8 + 16j, -9 - 14j, -9 + 14j],
        # [-8 - 16j, -8 + 16j, -9 - 14j, -9 + 14j, -10 - 12j, -10 + 12j],
        # [-9 - 14j, -9 + 14j, -10 - 12j, -10 + 12j, -11 - 10j, -11 + 10j],
        # [-2 - 25j, -2 + 25j, -3 - 20j, -3 + 20j, -4 - 15j, -4 + 15j],
        # [-3 - 30j, -3 + 30j, -4 - 25j, -4 + 25j, -5 - 20j, -5 + 20j],
        # [-4 - 35j, -4 + 35j, -5 - 30j, -5 + 30j, -6 - 25j, -6 + 25j],
        # [-5 - 40j, -5 + 40j, -6 - 35j, -6 + 35j, -7 - 30j, -7 + 30j],
        # [-6 - 45j, -6 + 45j, -7 - 40j, -7 + 40j, -8 - 35j, -8 + 35j],
        # [-7 - 50j, -7 + 50j, -8 - 45j, -8 + 45j, -9 - 40j, -9 + 40j],
        # [-8 - 55j, -8 + 55j, -9 - 50j, -9 + 50j, -10 - 45j, -10 + 45j],
        # [-9 - 60j, -9 + 60j, -10 - 55j, -10 + 55j, -11 - 50j, -11 + 50j],
        # [-10 - 65j, -10 + 65j, -11 - 60j, -11 + 60j, -12 - 55j, -12 + 55j],
        # [-11 - 70j, -11 + 70j, -12 - 65j, -12 + 65j, -13 - 60j, -13 + 60j]
    ]



    # desired_poles = [-3 - 26j, -3 + 26j, -4 - 24j, -4 + 24j, -5 - 22j, -5 + 22j]
    # desired_poles = [-3 - 26j, -3 + 26j, -4 - 24j, -4 + 24j, -5 - 22j, -5 + 22j]



    for i, polos in enumerate(polos_list):

        # print(polos)
        # quit()
        K = ctrl.place(A, B, polos)
        A_new = A - np.dot(B, K)

        sys_cl = ctrl.ss(A_new, B, C, D)
        G_cl = ctrl.ss2tf(sys_cl)

        plt.figure()
        ctrl.rlocus(G_cl)
        plt.title('Lugar das Raízes após Ajuste de Polos')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.grid(True)
        plt.show()

        #################### Malha Fechada #############
        velocidade_kmh = 80  # km/h
        velocidade_ms = velocidade_kmh / 3.6  # m/s
        raio = 300  # m
        aceleracao_centripeta = (velocidade_ms ** 2) / raio

        Mt = M_vazio + m0 + m1 + m2 + m3
        hcm_t = 1
        hcm_0 = 1 + h0
        hcm_1 = 1 + h1
        hcm_2 = 1 + h2
        hcm_3 = 1 + h3

        Hcm = (hcm_t*M_vazio + hcm_0*m0 + hcm_1*m1 + hcm_2*m2 + hcm_3*m3)/Mt
        Fc = (Mt) * aceleracao_centripeta
        My = 2*Hcm*Fc

        t, y = ctrl.step_response(My * sys_cl)

        plt.figure()
        plt.plot(t, y)
        plt.title('Resposta ao Degrau do Sistema Controlado')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Resposta (unidades)')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
