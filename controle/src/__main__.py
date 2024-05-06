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

    # Parâmetros iniciais
    m1 = 556.454
    m2 = 121.554
    m0 = 15408.992
    k1 = 192842.44
    k2 = 71003.48
    omega1 = 18.616
    omega2 = 24.169
    c1 = 6.289
    c2 = 1.784
    h1 = -0.1185548
    h2 = 0.03291344
    h0 = 1.4996e-6
    l = 2.6
    I0 = 647.31
    c_d = 62.524
    k = 5400
    g = 9.8
    If = -m0 * h0**2 + I0 + m1 * h1**2 + m2 * h2**2

    # Matrizes do sistema
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [-k1/m1, 0, g, -c1/m1, 0, 0],
                  [0, -k2/m2, g, 0, -c2/m2, 0],
                  [g*m1/If, g*m2/If, -k*l**2/(2*If), 0, 0, -c_d*l**2/(2*If)]])
    B = np.array([[0], [0], [0], [0], [0], [1/If]])
    C = np.array([[0, 0, 1, 0, 0, 0]])
    D = np.array([[0]])

    # Obtenção da função de transferência
    sys = ctrl.ss(A, B, C, D)
    G = ctrl.ss2tf(sys)

    # Simulação em malha aberta
    t, y = ctrl.step_response(G, T=np.linspace(0, 1.5, 100))
    plt.figure()
    # plt.figure(figsize=(14, 8))
    plt.plot(t, y)
    plt.title('Resposta ao degrau em malha aberta')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Resposta')
    plt.grid()
    plt.savefig(OUTPUT + 'degrau_em_malha_aberta' + '.png')
    plt.show()

    # Alocação de Polos
    polos = [-3.5 - 45j, -3.5 + 45j, -3 - 48j, -3 + 48j, -5 - 35j, -5 + 35j]

    K = ctrl.place(A, B, polos)
    F = A - B.dot(K)

    sys_closed = ctrl.ss(F, B, C, D)

    # Simulação em malha fechada
    t, y = ctrl.step_response(sys_closed, T=np.linspace(0, 1.5, 100))
    plt.figure()
    plt.plot(t, y)
    plt.title('Resposta ao degrau em malha fechada')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Resposta')
    plt.grid()
    plt.savefig(OUTPUT + 'degrau_em_malha_fechada' + '.png')
    plt.show()

    # Polos e zeros
    plt.figure()
    ctrl.pzmap(sys, title='Polos e zeros da malha aberta')
    plt.grid()
    plt.savefig(OUTPUT + 'polos_zeros_malha_aberta' + '.png')
    plt.figure()
    plt.grid()
    ctrl.pzmap(sys_closed, title='Polos e zeros da malha fechada')
    plt.savefig(OUTPUT + 'polos_zeros_malha_fechada' + '.png')
    plt.show()

    # Controlador LQR
    Q = np.diag([1e3, 1e3, 1e7, 1, 1, 1])
    R = 0.0001
    K_lqr, _, _ = ctrl.lqr(A, B, Q, R)
    F_lqr = A - B.dot(K_lqr)
    sys_lqr = ctrl.ss(F_lqr, B, C, D)

    t, y = ctrl.step_response(sys_lqr, T=np.linspace(0, 1.5, 100))
    plt.figure()
    plt.plot(t, y)
    plt.title('Resposta ao degrau com LQR')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Resposta')
    plt.grid()
    plt.savefig(OUTPUT + 'resposta_ao_degrau_com_LQR' + '.png')
    plt.show()

    # plt.show()

    # Definição dos polos para alocação
    p1o = -9.5 - 45j
    p2o = np.conj(p1o)
    p3o = -9 - 48j
    p4o = np.conj(p3o)
    p5o = -11 - 35j
    p6o = np.conj(p5o)
    pobs = [p1o, p2o, p3o, p4o, p5o, p6o]

    # Calculando o ganho K usando o lugar dos polos
    result = place_poles(A.T, C.T, pobs)
    K = result.gain_matrix.T

    # Sistema do observador
    O = A - K.dot(C)

    # Definição do sistema para simulação
    def sys_obs_aloc(t, x):
        return O.dot(x)

    # Condições iniciais
    e0 = [1, 0, 1, 1, 0, 1]

    # Configurações de tempo para a simulação
    t0 = 0
    dt = 0.001
    tf = 1.5
    t = np.arange(t0, tf+dt, dt)
    u = np.zeros_like(t)  # Input é zero

    # Simulação do sistema
    sol = solve_ivp(sys_obs_aloc, [t0, tf], e0, t_eval=t, vectorized=True)

    # Plotagem do mapa de polos
    plt.figure()
    plt.scatter(np.real(pobs), np.imag(pobs))
    plt.title('Mapa de Polos')
    plt.xlabel('Real')
    plt.ylabel('Imaginário')
    plt.grid()
    plt.savefig(OUTPUT + 'mapa_de_polos' + '.png')
    plt.show()

    # Plotagem das respostas do estado
    plt.figure()
    plt.plot(t, sol.y[2, :])
    plt.title('Resposta do Estado')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Estado 3')
    plt.grid()
    plt.savefig(OUTPUT + 'resposta_do_estado' + '.png')

    plt.show()

    # Matriz de pesos Q para o LQR
    Q0 = np.diag([10000, 1, 100, 1, 1, 1])

    # Peso para o termo de controle no LQR
    P0 = 0.001

    # Resolvendo a equação de Riccati para encontrar a matriz de ganho LQR
    P = solve_continuous_are(A.T, C.T, Q0, R)
    # Kobs = np.linalg.inv(P0) * P.dot(C)

    # Transposta de Kobs para obter K_obs
    # K_obs = Kobs.T
    K_obs = P.dot(C.T) / R

    # Sistema do observador
    O = A - K_obs.dot(C)

    # Condições iniciais
    e0 = [1, 0, 1, 0, 0, 0]

    # Configurações de tempo para a simulação
    t0 = 0
    dt = 0.001
    tf = 1.5
    t = np.arange(t0, tf+dt, dt)
    u = np.zeros_like(t)  # Input é zero

    # Definição do sistema para simulação
    def sys_obs_lqr(t, x):
        return O.dot(x)

    # Simulação do sistema
    sol = solve_ivp(sys_obs_lqr, [t0, tf], e0, t_eval=t, vectorized=True)

    # Plotagem das respostas dos estados 1 e 2
    plt.figure()
    plt.plot(t, sol.y[0, :], label='Estado 1')
    plt.plot(t, sol.y[1, :], label='Estado 2')
    plt.title('Resposta dos Estados 1 e 2')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Estados')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT + 'resposta_do_estado_1_2' + '.png')
    plt.show()

    # Plotagem da resposta do estado 3
    # plt.figure()
    # plt.plot(t, sol.y[2, :])
    # plt.title('Resposta do Estado 3')
    # plt.xlabel('Tempo (s)')
    # plt.ylabel('Estado 3')
    # plt.grid()
    # plt.savefig(OUTPUT + 'resposta_do_estado_3' + '.png')
    # plt.show()

    plt.figure()
    mag, phase, freq = bode(G, dB=True)
    plt.savefig(OUTPUT + 'diagrama_de_bode' + '.png')
    plt.show()

    # Mapa de polos em malha aberta
    plt.figure()
    ctrl.pzmap(sys, title='Mapa de Polos em Malha Aberta', plot=True)
    plt.grid(True)
    plt.savefig(OUTPUT + 'mapa_de_polos_malha_aberta.png')
    plt.show()

    polos_aberta = ctrl.poles(sys)
    polos_alocacao = ctrl.poles(sys_closed)

    # Plotagem dos polos em um único gráfico
    plt.figure()
    plt.scatter(np.real(polos_aberta), np.imag(polos_aberta), color='blue', marker='o', label='Malha Aberta')
    plt.scatter(np.real(polos_alocacao), np.imag(polos_alocacao), color='red', marker='x', label='Alocação de Polos')
    plt.title('Comparação de Polos: Malha Aberta vs Alocação de Polos')
    plt.xlabel('Parte Real')
    plt.ylabel('Parte Imaginária')
    plt.axvline(x=0, color='k', lw=1)  # Adiciona linha vertical em x=0 para melhor visualização
    plt.axhline(y=0, color='k', lw=1)  # Adiciona linha horizontal em y=0
    plt.grid(True)
    plt.legend()
    plt.show()


    # plt.show()


if __name__ == "__main__":
    main()
