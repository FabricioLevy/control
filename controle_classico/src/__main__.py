import numpy as np
import control as ctrl
from scipy.signal import place_poles
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys, os
from functions import ControlFunctions
from control.matlab import bode
from control import ctrb, obsv
import pandas as pd
from scipy.optimize import minimize
import scipy.signal as signal
import matplotlib.pyplot as plt

def create_folder(path):

    if os.path.isdir(path):
       pass
    else:
        os.mkdir(path)

def main():
    fx = ControlFunctions()

    create_folder('../output/')
    create_folder('../input/')

    INPUT = '../input/'
    OUTPUT = '../output/'

    g = 9.8
    M_vazio = 29000
    L = 2.9
    C = 17
    H = 2
    rho_fluido = 715
    hf = 2 * 0.6
    mf = rho_fluido * L * C * hf

    m1 = fx.massa_i(mf, 1, hf, L)
    m2 = fx.massa_i(mf, 2, hf, L)
    m3 = fx.massa_i(mf, 3, hf, L)
    m0 = mf - m1 - m2 - m3

    k1 = fx.calcular_ki(hf, L, g, mf, 1)
    k2 = fx.calcular_ki(hf, L, g, mf, 2)

    h0 = fx.calcular_hi(L, hf, 0)
    h1 = fx.calcular_hi(L, hf, 1)
    h2 = fx.calcular_hi(L, hf, 2)
    h3 = fx.calcular_hi(L, hf, 3)

    I0 = fx.momento_inercia(M_vazio, L, H)
    If = I0 + m0 * h0**2 + m1 * h1**2 + m2 * h2**2 + m3 * h3**2

    omega1 = np.sqrt(k1 / m1)
    omega2 = np.sqrt(k2 / m2)

    zeta = ((6 * 10**(-6)) / (np.sqrt(g) * L**(3 / 2)))**(1 / 2)

    c1 = fx.calcular_ci(m1, omega1, zeta, 1)
    c2 = fx.calcular_ci(m2, omega2, zeta, 2)

    k = 15000
    cd = 120

    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [-k1 / m1, 0, g, -c1 / m1, 0, 0],
        [0, -k2 / m2, g, 0, -c2 / m2, 0],
        [g * m1 / If, g * m2 / If, -k * L**2 / (2 * If), 0, 0, -cd * L**2 / (2 * If)]
    ])
    B = np.array([[0], [0], [0], [0], [0], [1 / If]])
    C = np.array([[0, 0, 1, 0, 0, 0]])
    D = np.array([[0]])

    sys = ctrl.ss(A, B, C, D)
    G = ctrl.ss2tf(sys)
    print(G)

    # Ajustar o numerador para eliminar os termos indesejados
    num = G.num[0][0]
    den = G.den[0][0]

    # Eliminar termos pequenos no numerador
    num_adjusted = np.copy(num)
    num_adjusted[0] = 0
    num_adjusted[1] = 0
    num_adjusted[2] = 0

    G_adjusted = ctrl.TransferFunction(num_adjusted, den)
    print("Função de Transferência Ajustada:")
    print(G_adjusted)

    fx.plot_poles(G_adjusted, OUTPUT + 'polos_zeros_malha_aberta.png')

    poles = ctrl.poles(G_adjusted)
    zeros = ctrl.zeros(G_adjusted)

    poles_data = {'Parte Real': np.real(poles), 'Parte Imaginária': np.imag(poles)}
    poles_df = pd.DataFrame(poles_data)

    print("Tabela de Polos:")
    print(poles_df)

    char_poly = np.poly(A)
    print("Polinômio Característico do Sistema:", char_poly)

    routh_array = fx.routh_hurwitz(char_poly)
    print("Tabela de Routh-Hurwitz:")
    print(routh_array)

    stable = np.all(routh_array[:, 0] > 0)
    print(stable)

    velocidade_kmh = 50  # km/h
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
    print('Hcm', Hcm)
    Fc = (Mt) * aceleracao_centripeta
    print('Fc', Fc)
    My = 2 * Hcm * Fc
    print('My', My)

    step_amplitude = My

    G_scaled = G_adjusted * step_amplitude
    t, y = ctrl.step_response(G_scaled)

    plt.figure()
    plt.plot(t, y)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Resposta')
    plt.title('Resposta ao Degrau com Amplitude Ajustada')
    plt.grid()
    plt.savefig(OUTPUT + 'resposta_ajustada.png')
    plt.show()

    # Root Locus PID Design
    rlist, klist = ctrl.root_locus(G_adjusted, xlim=(-4, 1), ylim=(-6, 6), plot=False)
    
    # Placeholder for actual ITAE and Gain Limit PID design
    Kp_itae, Ki_itae, Kd_itae = 1, 1, 1  # Replace with actual ITAE calculations
    Kp_gain, Ki_gain, Kd_gain = 1, 1, 1  # Replace with actual gain limit calculations

    controllers = {
        "Root Locus": (Kp_itae, Ki_itae, Kd_itae),
        "ITAE": (Kp_itae, Ki_itae, Kd_itae),
        "Gain Limit": (Kp_gain, Ki_gain, Kd_gain)
    }

    for method, (Kp, Ki, Kd) in controllers.items():
        pid_controller = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
        sys_closed_loop = ctrl.feedback(pid_controller * G_adjusted)

        t, y = ctrl.step_response(sys_closed_loop)

        plt.figure()
        plt.plot(t, y)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Resposta')
        plt.title(f'Resposta ao Degrau - {method}')
        plt.grid()
        plt.savefig(OUTPUT + f'resposta_{method.lower().replace(" ", "_")}.png')
        plt.show()

        mag, phase, omega = ctrl.bode(sys_closed_loop)
        plt.figure()
        plt.semilogx(omega, mag)  # Bode magnitude plot
        plt.figure()
        plt.semilogx(omega, phase)  # Bode phase plot
        plt.tight_layout()
        plt.savefig(OUTPUT + f'bode_{method.lower().replace(" ", "_")}.png')
        plt.show()

        ctrl.nyquist(sys_closed_loop)
        plt.tight_layout()
        plt.savefig(OUTPUT + f'nyquist_{method.lower().replace(" ", "_")}.png')
        plt.show()

if __name__ == "__main__":
    main()