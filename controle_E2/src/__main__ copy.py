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

    g= 9.8
    
    M_vazio = 29000
    L = 2.9
    C = 17
    H = 2
 
    rho_fluido = 715
    hf = 2*0.6
    mf = rho_fluido*L*C*hf

    m0 = fx.massa_i(mf, 0, hf, L)
    m1 = fx.massa_i(mf, 1, hf, L)
    m2 = fx.massa_i(mf, 2, hf, L)
    m3 = fx.massa_i(mf, 3, hf, L)
    m0 = mf - m1 - m2
    print('m0', m0)
    print('m1', m1)
    print('m2', m2)
    print('m3', m3)


    k0 = fx.calcular_ki(hf, L, g, mf, 0)
    k1 = fx.calcular_ki(hf, L, g, mf, 1)
    k2 = fx.calcular_ki(hf, L, g, mf, 2)

    print('k0', k0)
    print('k1',k1)
    print('k2',k2)

    h0 = fx.calcular_hi(L, hf, 0)
    h1 = fx.calcular_hi(L, hf, 1)
    h2 = fx.calcular_hi(L, hf, 2)
    h3 = fx.calcular_hi(L, hf, 3)

    print('h0', h0)
    print('h1',h1)
    print('h2',h2)
    print('h3',h3)

    I0 = fx.momento_inercia(M_vazio, L, H)
    print('I0', I0)

    If = I0 + -m0*h0**2 + m1*h1**2 + m2*h2**2
    print('If', If)

    ####################

    omega1 = np.sqrt(k1/m1)
    omega2 = np.sqrt(k2/m2)
    print('omega1', omega1)
    print('omega2', omega2)

    zeta = ((6*10**(-6))/(np.sqrt(g)*L**(3/2)))**(1/2)
    print(zeta)

    c1 = fx.calcular_ci(m1, omega1, zeta, 1)
    c2 = fx.calcular_ci(m2, omega2, zeta, 1)
    print(c1)
    print(c2)

    k = 15000
    cd = 120
    
    # Matrizes do sistema
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [-k1/m1, 0, g, -c1/m1, 0, 0],
                  [0, -k2/m2, g, 0, -c2/m2, 0],
                  [g*m1/If, g*m2/If, -k*L**2/(2*If), 0, 0, -cd*L**2/(2*If)]])
    B = np.array([[0], [0], [0], [0], [0], [1/If]])
    C = np.array([[0, 0, 1, 0, 0, 0]])
    D = np.array([[0]])

    sys = ctrl.ss(A, B, C, D)
    G = ctrl.ss2tf(sys)
    print(G)

    num = G.num[0][0]
    den = G.den[0][0]
    print("Numerador:", num)
    print("Denominador:", den)

    fx.plot_poles(sys, OUTPUT + 'polos_zeros_malha_aberta.png')

    # Cálculo dos polos e zeros
    poles = ctrl.poles(sys)
    zeros = ctrl.zeros(sys)

    # Criação da tabela de polos
    poles_data = {'Parte Real': np.real(poles), 'Parte Imaginária': np.imag(poles)}
    poles_df = pd.DataFrame(poles_data)

    # Exibindo a tabela de polos
    print("Tabela de Polos:")
    print(poles_df)

    char_poly = np.poly(A)
    print("Polinômio Característico do Sistema:", char_poly)

    # Construindo a tabela de Routh
    routh_array = fx.routh_hurwitz(char_poly)
    print("Tabela de Routh-Hurwitz:")
    print(routh_array)

    # Verificando estabilidade
    stable = np.all(routh_array[:, 0] > 0)
    print(stable)

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
    hcm_3 = 1 + h3 ############# adicionar m3

    Hcm = (hcm_t*M_vazio + hcm_0*m0 + hcm_1*m1 + hcm_2*m2 + hcm_3*m3)/Mt
    print('Hcm', Hcm)
    Fc = (Mt)*aceleracao_centripeta
    My = 2*Hcm*Fc
    print('My', My)

    # Configurando a amplitude do degrau

    step_amplitude = My

    G_scaled = G * step_amplitude
    t, y = ctrl.step_response(G_scaled)

    y_degress = y*(180/np.pi)

    plt.figure()
    plt.plot(t, y_degress)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (graus)')
    plt.ylim(0, 90)
    plt.xlim(0, 1)
    plt.title('Resposta com Amplitude')
    plt.grid()
    plt.savefig(OUTPUT + 'resposta_em_malha_aberta.png')
    # plt.show()

    delta_t = fx.calcular_delta_t(A)
    phi = fx.matriz_transicao_estados(A, delta_t)
    print(phi)
    Gamma = fx.matriz_delta(A, delta_t)
    print(Gamma)

    # Bode
    plt.figure()
    w = np.logspace(-1.5,1,200)
    mag, phase, freq = bode(G, w, Hz=True, dB=True)
    plt.tight_layout()
    plt.savefig(OUTPUT + 'diagrama_bode.png')
    # plt.show()

    # Analise de Controlabilidade e Observabilidade
    ctrl_matrix = ctrb(A, B)
    obsv_matrix = obsv(A, C)

    controlable = np.linalg.matrix_rank(ctrl_matrix) == A.shape[0]
    observable = np.linalg.matrix_rank(obsv_matrix) == A.shape[0]

    print("O sistema é controlável?", controlable)
    print("O sistema é observável?", observable)

    #####################
  
    # Alocação de Polos

    # polos = [-3 - 26j, -3 + 26j, -4 - 24j, -4 + 24j, -5 - 22j, -5 + 22j],

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

    
    for idx, polos in enumerate(polos_list):
        # Calcular os ganhos K usando a alocação de polos
        K = ctrl.place(A, B, polos)

        print("Ganhos K calculados:", K)

        F = A - B.dot(K)

        sys_closed = ctrl.ss(F, B, C, D)
        Gf = ctrl.ss2tf(sys_closed)

        G_scaled_cl = Gf * step_amplitude
        tf, yf = ctrl.step_response(G_scaled_cl)

        y_degress_f = yf*(180/np.pi)
        print(polos)


        plt.figure()
        plt.plot(tf, y_degress_f)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Ângulo (graus)')
        # plt.ylim(0, 90)
        # plt.xlim(0, 1)
        plt.title('Resposta com Amplitude')
        plt.grid()
        plt.show()
        plt.savefig(OUTPUT + 'resposta_em_malha_fechado.png')

        fx.plot_poles_mult(sys, sys_closed, OUTPUT + 'polos_zeros_comp.png')

    # Polos desejados
    desired_poles = np.array([-3 - 26j, -3 + 26j, -4 + 24j, -4 - 24j, -5 + 22j, -5 - 22j])

    # Função para calcular os ganhos LQR e verificar os polos
    def calculate_lqr_poles(Q, R):
        try:
            K, _, _ = ctrl.lqr(A, B, Q, R)
            F_lqr = A - B @ K
            poles_closed = np.linalg.eigvals(F_lqr)
            return poles_closed, K
        except np.linalg.LinAlgError:
            return None, None

    # Ajustar manualmente as matrizes Q e R
    Q = np.diag([100, 100, 100, 1, 1, 1])  # Ajuste conforme necessário
    R = np.array([[0.1]])  # Ajuste conforme necessário

    # Calcular os polos e ganhos LQR
    poles, K_lqr = calculate_lqr_poles(Q, R)

    # Exibir os resultados
    print("Polos calculados:\n", poles)
    print("Ganhos K_lqr:\n", K_lqr)
    print("Matriz Q:\n", Q)
    print("Matriz R:\n", R)

    # Plotar a resposta ao degrau do sistema em malha fechada
    if poles is not None:
        F_lqr = A - B @ K_lqr
        sys_closed = ctrl.ss(F_lqr, B, C, D)
        t, y = ctrl.step_response(sys_closed)
        y = y*(180/np.pi)

        
        plt.figure()
        plt.plot(t, y)
        plt.title('Resposta ao degrau com LQR')
        plt.xlabel('Tempo (s)')
        # plt.ylim(0, 90)
        plt.ylabel('Resposta')
        plt.grid()
        plt.show()
    else:
        print("Erro no cálculo dos polos. Ajuste as matrizes Q e R.")
    # quit()


    ################################################

    # Initial Error
    e0 = [1, 0, 1, 1, 0, 1]

    # Poles for allocation
    p1o = -9.5 - 45j
    p2o = np.conj(p1o)
    p3o = -9 - 48j
    p4o = np.conj(p3o)
    p5o = -11 - 35j
    p6o = np.conj(p5o)
    pobs = [p1o, p2o, p3o, p4o, p5o, p6o]
    
    print(pobs)
    quit()

        # Poles for observer allocation

    # Poles for controller allocation (example)
    p1c = -3 + 26j
    p2c = np.conj(p1c)
    p3c = -4 + 24j
    p4c = np.conj(p3c)
    p5c = -5 + 22j
    p6c = np.conj(p5c)
    pctrl = [p1c, p2c, p3c, p4c, p5c, p6c]

    fx.plot_comparative_poles(A, B, C, D, pobs, pctrl)


    quit()

#    # Calculate observer gain matrix
#     ko = signal.place_poles(A.T, C.T, pobs).gain_matrix.T

#     # Adjust the dimensions to ensure correct multiplication
#     ko = ko.reshape(-1, 1)  # Ensure ko is a column vector of shape (6, 1)

#     # Perform matrix multiplication
#     O = A - np.dot(ko, C)

#     # Create state-space system
#     sys_obs_aloc = signal.StateSpace(O, B, C, D)

#     # Time settings
#     t0 = 0
#     dt = 0.001
#     tf = 1.5
#     t = np.arange(t0, tf, dt)

#     # Input (zero input)
#     u = np.zeros(len(t))

#     # Simulate observer response
#     _, ys, xs = signal.lsim(sys_obs_aloc, U=u, T=t, X0=e0)

#     # Convert state-space system to transfer function
#     num, den = signal.ss2tf(O, B, C, D)

#     # Plot poles and zeros
#     fig3 = plt.figure()
#     poles, zeros, _ = signal.tf2zpk(num, den)
#     plt.scatter(np.real(poles), np.imag(poles), marker='o', color='red', label='Poles')
#     plt.scatter(np.real(zeros), np.imag(zeros), marker='x', color='blue', label='Zeros')
#     plt.axhline(0, color='black', lw=0.5)
#     plt.axvline(0, color='black', lw=0.5)
#     plt.xlabel('Real')
#     plt.ylabel('Imaginary')
#     plt.title('Poles and Zeros')
#     plt.legend()
#     plt.grid()

#     # Plot state 3
#     fig4 = plt.figure()
#     plt.plot(t, xs[:, 2])
#     plt.title('State 3 Response')
#     plt.xlabel('Time (s)')
#     plt.ylabel('State 3')

#     plt.show()

    ################################################

    # Calculate observer gain matrix
    ko = signal.place_poles(A.T, C.T, pobs).gain_matrix.T

    # Adjust the dimensions to ensure correct multiplication
    ko = ko.reshape(-1, 1)  # Ensure ko is a column vector of shape (6, 1)

    # Perform matrix multiplication
    O = A - np.dot(ko, C)

    # Create state-space system for the observer
    sys_obs_aloc = signal.StateSpace(O, B, C, D)

    # Time settings
    t0 = 0
    dt = 0.001
    tf = 1.5
    t = np.arange(t0, tf, dt)

    # Input (zero input)
    u = np.zeros(len(t))

    # Simulate observer response
    _, ys, xs_obs = signal.lsim(sys_obs_aloc, U=u, T=t, X0=e0)

    # Simulate real system response (assuming initial condition x0 is the same as e0)
    sys_real = signal.StateSpace(A, B, C, D)
    _, _, xs_real = signal.lsim(sys_real, U=u, T=t, X0=e0)

    # Calculate observation error
    error = xs_real - xs_obs

    # Plot observation error for each state
    # fig_error = plt.figure(figsize=(10, 8))
    # for i in range(A.shape[0]):
    #     plt.plot(t, error[:, i], label=f'State {i+1}')
    #     plt.title('Observation Error for Each State')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Error')
    #     plt.legend()
    #     plt.grid()
    # quit()

    fig_error = plt.figure(figsize=(10, 8))
    for i in [2]:  # Only plotting states 1 and 2 (index 0 and 1)
        plt.plot(t, error[:, i], label=f'X{i+1}')
    plt.title('Erro de Observação para as Coordenadas Modais')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro X3')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT + 'err_observacao_x3')
    # quit()


    fig_error = plt.figure(figsize=(10, 8))
    for i in [0, 1]:  # Only plotting states 1 and 2 (index 0 and 1)
        plt.plot(t, error[:, i], label=f'X{i+1}')
    plt.title('Erro de Observação para as Coordenadas Modais')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro X1 e X2')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT + 'err_observacao_x1_x2')
    plt.show()
    # quit()

    # Plot observation error for state 3 (x3)
    # fig_error = plt.figure(figsize=(10, 6))
    # plt.plot(t, error[:, 2], label='State 3 (x3) Error')
    # plt.title('Observation Error for Position Angular x3')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.grid()

    # Plot poles and zeros
    fig3 = plt.figure()
    num, den = signal.ss2tf(O, B, C, D)
    poles, zeros, _ = signal.tf2zpk(num, den)
    plt.scatter(np.real(poles), np.imag(poles), marker='o', color='red', label='Poles')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='x', color='blue', label='Zeros')
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Poles and Zeros')
    plt.legend()
    plt.grid()

    # Plot state 3
    fig4 = plt.figure()
    plt.plot(t, xs_obs[:, 2], label='Observer')
    plt.plot(t, xs_real[:, 2], label='Real')
    plt.title('State 3 Response')
    plt.xlabel('Time (s)')
    plt.ylabel('State 3')
    plt.legend()
    plt.grid()

    plt.show()





if __name__ == "__main__":
    main()
