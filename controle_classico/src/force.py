import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
from functions import ControlFunctions

def main():
    fx = ControlFunctions()

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

    m0 = fx.massa_i(mf, 0, hf, L)
    m1 = fx.massa_i(mf, 1, hf, L)
    m2 = fx.massa_i(mf, 2, hf, L)
    m3 = fx.massa_i(mf, 3, hf, L)
    m0 = mf - m1 - m2

    k0 = fx.calcular_ki(hf, L, g, mf, 0)
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

    zeta = ((6 * 10**(-6)) / (np.sqrt(g) * L**(3 / 2)))**(1 / 2)

    c1 = fx.calcular_ci(m1, omega1, zeta, 1)
    c2 = fx.calcular_ci(m2, omega2, zeta, 1)

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

    # Resolução do Nx e Nu para controle de referência
    A1 = np.block([
        [A, B],
        [C, D]
    ])
    A2 = np.linalg.inv(A1)
    No = np.zeros((A1.shape[0], 1))
    # No[-1, 0] = 1
    No[-1, 0] = 0.174*180/np.pi
    Nxu = A2 @ No

    Nx = Nxu[:A.shape[0]]
    Nu = Nxu[A.shape[0]:]

    print(Nx)
    print(Nu)

    # Verificação das dimensões de Nx e Nu
    print(f'Dimensions of Nx: {Nx.shape}')
    print(f'Dimensions of Nu: {Nu.shape}')

    # Definindo os polos desejados
    p1o = -9.5 - 45j
    p2o = np.conj(p1o)
    p3o = -9 - 48j
    p4o = np.conj(p3o)
    p5o = -11 - 35j
    p6o = np.conj(p5o)
    p = [p1o, p2o, p3o, p4o, p5o, p6o]

    # p1o = -3 - 26j
    # p2o = np.conj(p1o)
    # p3o = -4 - 24j
    # p4o = np.conj(p3o)
    # p5o = -5 - 22j
    # p6o = np.conj(p5o)
    # p = [p1o, p2o, p3o, p4o, p5o, p6o]


    # p1c = -3 + 26j
    # p2c = np.conj(p1c)
    # p3c = -4 + 24j
    # p4c = np.conj(p3c)
    # p5c = -5 + 22j
    # p6c = np.conj(p5c)
    # pctrl = [p1c, p2c, p3c, p4c, p5c, p6c]

    # Calculando a matriz de ganho K
    K = signal.place_poles(A, B, p).gain_matrix

    # Calculando F
    F = A - B @ K

    # Sistema de espaço de estados com o seguidor de referência
    B2 = B @ Nu + B @ K @ Nx
    C1 = C

    # Verificando as dimensões antes de criar o sistema
    print(f'Dimensions of F: {F.shape}')
    print(f'Dimensions of B2: {B2.shape}')
    print(f'Dimensions of C1: {C1.shape}')
    print(f'Dimensions of D: {D.shape}')
    # quit()

    # Criação do sistema de espaço de estados
    sys = signal.StateSpace(F, B2, C1, D)

    # Simulação do sistema
    # t = np.linspace(0, 10, 1000)
    t = np.linspace(0, 2, 1000)
    t, y, x = signal.lsim(sys, U=np.ones_like(t), T=t)  # U=np.ones_like(t) para referência de unidade

    # Plot dos resultados
    plt.figure()
    plt.plot(t, y, label='Saída do Sistema')
    plt.axhline(y=0.174*180/np.pi, color='r', linestyle='--', label='Referência')
    plt.title('Seguidor de Referência para $\psi$')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (graus)')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT + 'seguidor_referencia_psi.png')
    plt.show()

    # Calcular a força nos atuadores
    force_actuators = -K @ x.T

    # Plotar a força nos atuadores
    plt.figure(figsize=(10, 8))
    plt.plot(t, force_actuators.T, label=['Força nos Atuadores'])
    plt.xlabel('Tempo (s)')
    plt.ylabel('Força (N)')
    plt.title('Força nos Atuadores ao Longo do Tempo')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT + 'force_actuators.png')
    plt.show()

    quit()

if __name__ == "__main__":
    main()
