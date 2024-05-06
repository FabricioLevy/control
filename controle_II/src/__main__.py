import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import sys, os
from functions import ControlFunctions
import scipy.linalg as la
from control import ctrb, obsv
from scipy.special import gamma
from scipy.linalg import fractional_matrix_power

def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def main():

    fx = ControlFunctions()

    create_folder('../output/')
    create_folder('../input/')

    INPUT = '../input/'
    OUTPUT = '../output/'

    # Parâmetros iniciais
    g = 9.8  # gravidade [m/s^2]
    hf = 1.7  # altura do fluido [m]
    l = 19.28  # comprimento médio de um vagão [m]
    m_d = 132564  # massa de diesel transportado dentro do tanque [kg]
    n = 2  # modos

    # Inicialização dos arrays
    m = np.zeros(n)
    k = np.zeros(n)
    omega_n = np.zeros(n)  # frequência natural do sistema
    c = np.zeros(n)

    viscosidade_cinematica = 2.5e-5  # viscosidade cinemática do combustível diesel a 20ºC [m^2/s]
    n = 2  # modos

    # Cálculo do Stigma
    Stigma = np.sqrt(viscosidade_cinematica / (l ** (3 / 2) * np.sqrt(g)))

    # Cálculos usando métodos da classe ControlFunctions
    for i in range(n):
        m[i] = fx.massa_i(m_d, i, hf, l)
        k[i] = fx.rigidez_i(hf, m_d, g, l, i)
        omega_n[i] = np.sqrt(k[i] / m[i])
        c[i] = fx.amortecimento_i(m[i], omega_n[i], Stigma)

    m0 = fx.massa_0(m, m_d)  # [kg]
    Mt = 130000  # [kg]
    Kt = 3848000  # [N/m]
    Ct = 1414553  # [kg/s]

    # Matrizes de espaço de estados
    m1, m2 = m[0], m[1]
    k1, k2 = k[0], k[1]
    c1, c2 = c[0], c[1]

    A3 = np.array([
        [-k1/m1, 0, 0, 0, k1/m1, 0],
        [0, -k2/m2, 0, 0, k2/m2, 0],
        [0, 0, -k1/m1, 0, 0, k1/m1],
        [0, 0, 0, -k2/m2, 0, k2/m2],
        [k1/(Mt+m0), k2/(Mt+m0), 0, 0, -(Kt+k1+k2)/(Mt+m0), Kt/(Mt+m0)],
        [0, 0, k1/(Mt+m0), k2/(Mt+m0), Kt/(Mt+m0), -(Kt+k1+k2)/(Mt+m0)]
    ])

    A4 = np.array([
        [-c1/m1, 0, 0, 0, c1/m1, 0],
        [0, -c2/m2, 0, 0, c2/m2, 0],
        [0, 0, -c1/m1, 0, 0, c1/m1],
        [0, 0, 0, -c2/m2, 0, c2/m2],
        [c1/(Mt+m0), c2/(Mt+m0), 0, 0, -(Ct+c1+c2)/(Mt+m0), Ct/(Mt+m0)],
        [0, 0, c1/(Mt+m0), c2/(Mt+m0), Ct/(Mt+m0), -(Ct+c1+c2)/(Mt+m0)]
    ])

    A = np.block([
        [np.zeros((6, 6)), np.eye(6)],
        [A3, A4]
    ])
    B = np.vstack([np.zeros((6, 1)), [[0], [0], [0], [0], [630976.9/(Mt+m0)], [0]]])
    C = np.array([[0, 0, 0, 0, Kt, -Kt, 0, 0, 0, 0, Ct, -Ct]])
    D = np.array([[0]])

    # Intervalo de simulação
    t0 = 0
    tfinal = 20
    passos = 1000
    t = np.linspace(t0, tfinal, passos)
    dt = (tfinal - t0) / passos

    # Duplo degrau
    Ft = np.zeros(passos)
    for i in range(50, passos // 2 + 50):
        Ft[i] = 630976.9
    for i in range(passos // 2 + 50, passos):
        Ft[i] = -630976.9

    # Plot do sinal de entrada
    plt.plot(t, Ft)
    plt.title('Entrada Duplo Degrau')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Força (N)')
    plt.savefig(OUTPUT + 'entrada_duplo_degrau.png')
    # plt.show()

   # Analise de Controlabilidade e Observabilidade
    ctrl_matrix = ctrb(A, B)
    obsv_matrix = obsv(A, C)

    controlable = np.linalg.matrix_rank(ctrl_matrix) == A.shape[0]
    observable = np.linalg.matrix_rank(obsv_matrix) == A.shape[0]

    print("O sistema é controlável?", controlable)
    print("O sistema é observável?", observable)

    # Condições iniciais
    x0 = np.zeros((12,))

    # Criar o sistema linear
    sistema_lin = ctrl.ss(A, B, C, D)

    # Função de Transferência
    G_FT = ctrl.ss2tf(sistema_lin)

    # Mostrar zeros e polos
    zeros = ctrl.zeros(G_FT)
    polos = ctrl.poles(G_FT)

    print("Zeros:")
    print(zeros)
    print("Polos:")
    print(polos)

    # Verificar estabilidade
    estavel = all(p.real < 0 for p in polos)
    print("O sistema é estável?" if estavel else "O sistema é instável.")

    #################################################################

    t = np.linspace(0, 20, 1000)
    dt = t[1] - t[0]  # Intervalo de tempo entre os pontos
    Ft = np.zeros_like(t)  # Força aplicada
    Ft[50:501] = 630976.9  # Definição do primeiro degrau
    Ft[501:] = -630976.9   # Definição do segundo degrau

    xt = [x0.reshape(-1, 1)]

    # Simulação do sistema
    for i in range(1, len(t)):
        B_aux = np.vstack([np.zeros((6, 1)), np.array([[0, 0, 0, 0, Ft[i], 0]]).T])
        x_aux = fx.OMEGA(A, B_aux, 1/(Mt + m0), dt, 4, xt[-1])
        xt.append(x_aux)

    xt = np.column_stack(xt)

    # Criação do sistema linear e cálculo de polos e zeros
    sistema_lin = ctrl.ss(A, B, C, D)
    G_FT = ctrl.ss2tf(sistema_lin)

    # Gráfico de Polos e Zeros
    plt.figure()
    ctrl.pzmap(G_FT, plot=True)  # Corrigido para plot=True
    plt.title('Mapa de Polos e Zeros')
    plt.savefig(OUTPUT + 'mapa_de_polos_e_zeros.png')

    # Plotagem dos resultados
    plt.figure()
    plt.plot(t, xt[4, :] - xt[5, :], label='Posição Relativa x1')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição (m)')
    plt.title('Evolução da Posição Relativa x1')
    plt.legend()
    plt.savefig(OUTPUT + 'evolução_da_posicao_relativa_x1.png')
    plt.show()
    
    # Criação do sistema de controle linear
    sistema_lin = ctrl.ss(A, B, C, D)

    # Conversão para função de transferência
    G_FT = ctrl.ss2tf(sistema_lin)

    # Gráfico de Bode
    mag, phase, omega = ctrl.bode(G_FT, np.logspace(-1, 1, 100), plot=True)
    plt.savefig(OUTPUT + 'bode.png')

    # Mostra o gráfico
    plt.show()

    Q = np.eye(12) #ajuste conforme o número de estados e suas prioridades
    R = np.array([[1]])  # Ponderação para o input

    # Computa o ganho do controlador LQR
    K, S, E = ctrl.lqr(A, B, Q, R)

    # Imprime o ganho e os polos do sistema fechado
    print("Ganho LQR K:", K)
    print("Polos do sistema fechado:", E)

    # Você pode querer verificar o sistema fechado
    A_cl = A - B.dot(K)
    print("Matriz do sistema fechado A_cl:", A_cl)


    # Sistema de controle fechado
    A_cl = A - B.dot(K)
    sistema_fechado = ctrl.ss(A_cl, B, np.eye(12), np.zeros((12, 1)))

    # Simulação de resposta ao degrau
    T = np.linspace(0, 10, 100)  # 10 segundos, 100 pontos de tempo
    T, Y = ctrl.step_response(sistema_fechado, T)

    # Ajustando a forma de Y se necessário
    if Y.ndim > 2:  # Y pode ser (output, time, input)
        Y = Y.squeeze()  # Remove dimensões singulares

    # Plotando a resposta ao degrau para cada estado
    plt.figure(figsize=(12, 10))
    for i in range(12):
        plt.subplot(4, 3, i+1)
        plt.plot(T, Y[i], label=f'Estado {i+1}')  # Ajuste aqui para Y[i]
        plt.xlabel('Tempo (s)')
        plt.ylabel(f'x{i+1}')
        plt.title(f'Resposta do Estado {i+1}')
        plt.grid(True)
        plt.tight_layout()
    plt.savefig(OUTPUT + f'resposta_do_estado{i+1}.png')
    plt.show()

    # Polos do sistema aberto
    polos_abertos = np.linalg.eigvals(A)

    # Sistema de controle fechado
    A_cl = A - B.dot(K)
    polos_fechados = np.linalg.eigvals(A_cl)

    # Plotando os polos no plano complexo
    plt.figure(figsize=(8, 8))
    plt.scatter(polos_abertos.real, polos_abertos.imag, color='red', marker='x', label='Polos Originais')
    plt.scatter(polos_fechados.real, polos_fechados.imag, color='blue', marker='o', label='Polos com LQR')
    plt.title('Comparação dos Polos do Sistema')
    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginária')
    plt.axvline(0, color='gray', lw=1)  # Eixo Y do plano complexo
    plt.axhline(0, color='gray', lw=1)  # Eixo X do plano complexo
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Mantém a mesma escala em ambos os eixos
    plt.savefig(OUTPUT + f'comparacao_dos_polos_do_sistema.png')
    plt.show()



if __name__ == '__main__':
    main()