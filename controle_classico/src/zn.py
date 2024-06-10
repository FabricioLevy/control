import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
from functions import ControlFunctions
import matplotlib.ticker as mticker

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

    sys_ss = ctrl.ss(A, B, C, D)
    sys_tf = ctrl.ss2tf(sys_ss)
    print("Função de Transferência do Sistema:")
    print(sys_tf)

    poles = np.linalg.eigvals(A)
    zeros = np.linalg.eigvals(B.T @ np.linalg.pinv(A) @ C.T)
    print(poles)
    print(zeros)

    # Ganho crítico e período crítico (assumido)
    K_cr = 40
    # K_cr_values = [20, 25, 30, 35, 40]
    P_cr = 2 * np.pi
    # print(K_cr)
    print(P_cr)
    # quit()

    # Cálculo dos parâmetros PID usando Ziegler-Nichols
    # K_p = 0.5 * K_cr
    # T_i = 0.4 * P_cr
    # T_d = 0.05 * P_cr

    K_p = 0.5 * K_cr
    T_i = 0.4 * P_cr
    T_d = 0.05 * P_cr

    print("Parâmetros do Controlador PID Ajustados:")
    print(f"K_p = {K_p}")
    print(f"T_i = {T_i}")
    print(f"T_d = {T_d}")

    # Função de transferência do sistema
    num = [1]
    den = [1, 6, 5, 0]  # Denominador conforme a imagem (s^3 + 6s^2 + 5s)
    sys = ctrl.TransferFunction(num, den)

    # Função de transferência do controlador PID
    K_i = K_p / T_i
    K_d = K_p * T_d
    pid_num = [K_d, K_p, K_i]
    pid_den = [1, 0]
    print('pid_num', pid_num)
    print('pid_den', pid_den)
    # quit()
    controller = ctrl.TransferFunction(pid_num, pid_den)

    # Função de transferência em malha aberta
    open_loop_tf = controller * sys

    # Plot do root locus
    plt.figure()
    ctrl.rlocus(open_loop_tf, plot=True)
    plt.title("Root Locus do Sistema com Controlador PID")
    plt.xlabel("Parte Real")
    plt.ylabel("Parte Imaginária")
    plt.grid(True)
    plt.show()

    # Malha fechada do sistema com o controlador PID
    system_closed_loop = ctrl.feedback(controller * sys)

    # Resposta ao degrau
    t, y = ctrl.step_response(system_closed_loop, T=10)  # Simulação para 10 segundos

    # Plot da resposta ao degrau
    plt.figure()
    plt.plot(t, y)
    plt.title("Resposta ao Degrau do Sistema com Controlador PID por ganho limite")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Ângulo [graus]")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
