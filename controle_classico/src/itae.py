import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.optimize import least_squares

def itae_pid_tuning(sys_tf, wn_values):
    s = ctrl.TransferFunction.s
    t = np.linspace(0, 0.5, 1000)  # Tempo de simulação

    plt.figure()
    for wn in wn_values:
        # Coeficientes do polinômio ITAE para ordem 7
        itae = s**7 + 2.217*wn*s**6 + 6.745*wn**2*s**5 + 9.349*wn**3*s**4 + 11.58*wn**4*s**3 + 8.68*wn**5*s**2 + wn**6*s

        pid_controller = itae / (s**2 + 2*0.7*wn*s + wn**2)
        sys_closed_loop = ctrl.feedback(pid_controller * sys_tf, 1)

        t, y = ctrl.step_response(sys_closed_loop, T=t)
        y = -0.1 * y  # Ajustar para estabilizar em 0.5 rad
        plt.plot(t, y, label=f'wn={wn} rad/s')

    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo psi (rad)')
    plt.title('Sintonia ITAE para diferentes valores de wn')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    OUTPUT = '../output/'

    # Definindo as matrizes conforme seu problema específico
    g = 9.8
    M_vazio = 29000
    L = 2.9
    C = 17
    H = 2
    rho_fluido = 715
    hf = 2 * 0.6
    mf = rho_fluido * L * C * hf

    m0 = 0.25 * mf
    m1 = 0.25 * mf
    m2 = 0.25 * mf
    m3 = 0.25 * mf

    k0 = 15000
    k1 = 15000
    k2 = 15000

    h0 = 0
    h1 = 0.6
    h2 = 1.2
    h3 = 1.8

    I0 = M_vazio * (L**2 + H**2) / 12
    If = I0 + m0 * h0**2 + m1 * h1**2 + m2 * h2**2 + m3 * h3**2

    omega1 = np.sqrt(k1 / m1)
    omega2 = np.sqrt(k2 / m2)

    zeta = ((6 * 10**(-6)) / (np.sqrt(g) * L**(3 / 2)))**(1 / 2)

    c1 = 2 * zeta * omega1 * m1
    c2 = 2 * zeta * omega2 * m2

    k = 15000
    cd = 120

    # Matrizes do sistema
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [-k1 / m1, 0, g, -c1 / m1, 0, 0],
        [0, -k2 / m2, g, 0, -c2 / m2, 0],
        [g * m1 / If, g * m2 / If, -k * L**2 / (2 * If), 0, 0, -cd * L**2 / (2 * If)]
    ])
    B = np.array([[0], [0], [0], [0], [0], [1 / If]])
    C = np.array([[0, 0, 1, 0, 0, 0]])  # Saída para psi
    D = np.array([[0]])

    # Criação do sistema de espaço de estados
    sys_ss = ctrl.ss(A, B, C, D)

    # Conversão para função de transferência
    sys_tf = ctrl.ss2tf(sys_ss)
    print("Função de Transferência do Sistema:")
    print(sys_tf)

    # Verificar os polos e zeros do sistema em malha aberta
    poles = np.linalg.eigvals(A)
    zeros = np.linalg.eigvals(B.T @ np.linalg.pinv(A) @ C.T)  # Usar pinv para a pseudoinversa
    print("Polos do Sistema em Malha Aberta:", poles)
    print("Zeros do Sistema em Malha Aberta:", zeros)

    # Valores de wn para sintonia ITAE
    wn_values = np.arange(30, 35, 1)

    # Sintonia ITAE e plot
    itae_pid_tuning(sys_tf, wn_values)

if __name__ == "__main__":
    main()
