import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import sys, os
from functions import ControlFunctions
import scipy.linalg as la

def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def main():
    fx = ControlFunctions()

    create_folder('../output/')
    create_folder('../input/')

    g = 9.81  # Aceleração devido à gravidade
    hf = 1.7  # Altura do fluido
    l = 19.28  # Comprimento médio de um vagão
    m_d = 132564  # Massa de Diesel
    visc_cin = 2.5 * 10**(-5)
    n = 2
    Mt = 130000  # Massa total do sistema
    Kt = 3848000  # Rigidez total
    Ct = 1414553  # Amortecimento total
    F = 630976.9  # Força aplicada

    m, k, c, m0 = fx.calcular_parametros(m_d, hf, l, g, n, Mt, Kt, Ct, visc_cin)
    A = fx.criar_matriz_A(m, k, c, m0, Mt, Kt, Ct)
    b111 = F / (Mt + m0)
    B = np.zeros((12, 1))
    B[10] = b111

    C = np.zeros((1, 12))
    C[0, 4] = Kt
    C[0, 5] = -Kt
    C[0, 10] = Ct
    C[0, 11] = -Ct

    D = np.zeros((1, 1))

    sistema = ctrl.ss(A, B, C, D)
    G = ctrl.ss2tf(sistema)

    # Análise de Controlabilidade
    Cm = ctrl.ctrb(A, B)
    controlabilidade_rank = np.linalg.matrix_rank(Cm)
    print("Rank da Matriz de Controlabilidade:", controlabilidade_rank, "de", A.shape[0])
    
    if controlabilidade_rank == A.shape[0]:
        print("O sistema é completamente controlável.")
    else:
        print("O sistema não é completamente controlável.")

    # Simulação em malha aberta
    t, y = ctrl.step_response(G, T=np.linspace(0, 1.5, 100))
    plt.figure()
    plt.plot(t, y)
    plt.title('Resposta ao degrau em malha aberta')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Resposta')
    plt.savefig('../output/degrau_em_malha_aberta.png')
    plt.show()

    # sys = ctrl.StateSpace(A, B, C, D)
    sys = ctrl.ss(A, B, C, D)

    # Conversão para função de transferência
    G_FT = ctrl.ss2tf(sys)

    # # Obtendo numerador e denominador da função de transferência
    num, den = G_FT.num[0][0], G_FT.den[0][0]

    # # Cálculo dos zeros e polos
    zeros = np.roots(num)
    polos = np.roots(den)
    print(zeros)
    print(polos)

if __name__ == "__main__":
    main()
