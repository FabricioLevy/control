import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
from functions import ControlFunctions
import matplotlib.ticker as mticker
import pandas as pd
from scipy.linalg import schur

def schur_truncation(sys, order):
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    
    # Decomposição de Schur
    T, Z = schur(A, output='complex')
    
    # Truncando os estados menos significativos
    Ar = T[:order, :order]
    Br = Z[:, :order].T @ B
    Cr = C @ Z[:, :order]
    Dr = D
    
    return ctrl.ss(Ar, Br, Cr, Dr)

def itae_pid_tuning(wn, zeta=0.7):
    # Parâmetros de sintonia baseados no critério ITAE ajustados para sistemas oscilatórios
    Kp = (0.8 * wn**2)
    Ki = (1.2 * wn)
    Kd = (0.45 / wn)

    pid = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    return pid

def main():
    fx = ControlFunctions()

    OUTPUT = '../output/'

    g = 9.8
    M_vazio = 29000
    L = 2.9
    C = 17
    H = 2
    rho_fluido = 715
    hf = 2 * 0.6
    mf = rho_fluido * L * C * hf

    m0 = fx.massa_i(mf, 0, hf, L)
    m1 = fx.massa_i(mf, 1, hf, L)
    m2 = fx.massa_i(mf, 2, hf, L)
    m3 = fx.massa_i(mf, 3, hf, L)
    m0 = mf - m1 - m2 - m3

    k0 = fx.calcular_ki(hf, L, g, mf, 0)
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
    c2 = fx.calcular_ci(m2, omega2, zeta, 1)

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
    print('Hcm', Hcm)
    Fc = Mt * aceleracao_centripeta
    print('Fc', Fc)
    My = 2 * Hcm * Fc
    print('My', My)

    sys = ctrl.ss(A, B, C, D)

    G = ctrl.ss2tf(sys)

    G_scaled = G * My

    num = G.num[0][0]
    den = G.den[0][0]

    # Reduzindo a ordem do sistema
    reduced_order = 2  # Defina a ordem desejada
    system_reduced = schur_truncation(sys, reduced_order)

    # Convertendo de volta para a função de transferência e aplicando o fator My
    G_reduced = ctrl.ss2tf(system_reduced) * My
    num_reduced = G_reduced.num[0][0]
    den_reduced = G_reduced.den[0][0]

    print("Função de transferência original: Numerador:", num, "Denominador:", den)
    print("Função de transferência reduzida: Numerador:", num_reduced, "Denominador:", den_reduced)

    # Valores de wn para teste
    wn_values = [10, 11, 12, 13, 14, 15]

    # Plotando a resposta ao degrau para diferentes valores de wn
    t = np.linspace(0, 100, 1000)
    plt.figure()
    for wn in wn_values:
        pid = itae_pid_tuning(wn)
        G_closed_loop = ctrl.feedback(pid * G_reduced)
        t_reduced, y_reduced = ctrl.step_response(G_closed_loop, T=t)
        plt.plot(t_reduced, y_reduced, label=f'Sistema Reduzido com PID (ITAE) wn={wn}', linestyle='--')

    t_original, y_original = ctrl.step_response(G_scaled, T=t)
    plt.plot(t_original, y_original, label='Sistema Original')
    plt.title('Resposta para Diferentes Valores de wn')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo psi (rad)')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotando a resposta ao impulso para diferentes valores de wn
    plt.figure()
    for wn in wn_values:
        pid = itae_pid_tuning(wn)
        G_closed_loop = ctrl.feedback(pid * G_reduced)
        t_reduced_impulse, y_reduced_impulse = ctrl.impulse_response(G_closed_loop, T=t)
        plt.plot(t_reduced_impulse, y_reduced_impulse, label=f'Sistema Reduzido com PID (ITAE) wn={wn}', linestyle='--')

    t_original_impulse, y_original_impulse = ctrl.impulse_response(G_scaled, T=t)
    plt.plot(t_original_impulse, y_original_impulse, label='Sistema Original')
    plt.title('Resposta ao Impulso para Diferentes Valores de wn')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo psi (rad)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
