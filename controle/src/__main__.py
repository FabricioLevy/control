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
        # plt.show()
        plt.savefig(OUTPUT + 'resposta_em_malha_fechado.png')

        fx.plot_poles_mult(sys, sys_closed, OUTPUT + 'polos_zeros_comp.png')
    
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

    p1c = -3 + 26j
    p2c = np.conj(p1c)
    p3c = -4 + 24j
    p4c = np.conj(p3c)
    p5c = -5 + 22j
    p6c = np.conj(p5c)
    pctrl = [p1c, p2c, p3c, p4c, p5c, p6c]

    fx.plot_comparative_poles(A, B, C, D, pobs, pctrl, OUTPUT + 'polos_obs_contr.png')


    # Create state-space system for the observer
    ko = signal.place_poles(A.T, C.T, pobs).gain_matrix.T
    ko = ko.reshape(-1, 1)  # Ensure ko is a column vector of shape (6, 1)
    O = A - np.dot(ko, C)
    sys_obs_aloc = signal.StateSpace(O, B, C, D)

    # Time settings
    t0 = 0
    dt = 0.001
    tf = 1.5
    t = np.arange(t0, tf, dt)

    # Input (zero input)
    u = np.zeros(len(t))

    # e0 = np.full(6, 20)
    e0 = np.full(A.shape[0], 20)


    # Simulate observer response
    _, ys, xs_obs = signal.lsim(sys_obs_aloc, U=u, T=t, X0=e0)

    # Simulate real system response (assuming initial condition x0 is the same as e0)
    sys_real = signal.StateSpace(A, B, C, D)
    _, _, xs_real = signal.lsim(sys_real, U=u, T=t, X0=e0)

    # Calculate observation error
    error = xs_obs

    fig_error = plt.figure(figsize=(10, 8))
    for i in [0, 1]:  # Only plotting states 1 and 2 (index 0 and 1)
        plt.plot(t, error[:, i], label=f'X{i+1}')
    plt.title('Observação para as Coordenadas Modais')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro X1 e X2')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT + 'err_observacao_x1_x2')
    plt.show()

    fig_error = plt.figure(figsize=(10, 8))
    for i in [2]:  # Only plotting states 1 and 2 (index 0 and 1)
        plt.plot(t, error[:, i], label='$\psi$')
    plt.title('Observador para $\psi$')
    plt.xlabel('Tempo (s)')
    plt.ylabel('$\psi$')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT + 'err_observacao_x3')
    plt.show()
    ##################################################

    # Plotting
    fig_error = plt.figure(figsize=(10, 8))
    for i in [0, 1]:  # Only plotting states 1 and 2 (index 0 and 1)
        plt.plot(t, error[:, i], label=f'X{i+1}')
    plt.title('Erro de Observação para as Coordenadas Modais')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro X1 e X2')
    plt.legend()
    plt.grid()
    plt.show()
    # quit()

    # Calculate observer gain matrix
    L = signal.place_poles(A.T, C.T, pobs).gain_matrix.T

    # Calculate controller gain matrix
    K = signal.place_poles(A, B, pctrl).gain_matrix

    # Define the augmented state-space system
    A_aug = np.block([
        [A - B @ K, B @ K],
        [np.zeros_like(A), A - L @ C]
    ])

    B_aug = np.block([
        [B],
        [np.zeros_like(B)]
    ])
    C_aug = np.block([C, np.zeros_like(C)])
    D_aug = D

    sys_aug = signal.StateSpace(A_aug, B_aug, C_aug, D_aug)

    # Simulation parameters
    t0 = 0
    dt = 0.001
    tf = 10
    t = np.arange(t0, tf, dt)

    # Initial condition with a perturbation
    x0 = np.zeros(A_aug.shape[0])
    x0[2] = 0.0  # Perturbation in the third state


    # Input (zero input for simplicity)
    u = np.zeros(len(t))
    u[0] = 1
    u[2] = 1
    u[5] = 1
    u[8] = 1

    _, y, x_aug = signal.lsim(sys_aug, U=u, T=t, X0=x0)

    psi_est = x_aug[:, 8]

    psi_real = x_aug[:, 2]

    plt.figure()
    plt.plot(t, psi_real*(180/np.pi), label='Real $\psi$')
    plt.title('Ângulo Real $\psi$ pelo Princípio da Separação')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Real $\psi$ (Graus)')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT + 'real_phi_sep.png')
    plt.show()

    error_psi = psi_real - psi_est
    plt.figure()
    plt.plot(t, error_psi, label='Erro de $\psi$')
    plt.title('Erro Observador $\psi$')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro de $\psi$')
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT + 'erro_observacao_phi_sep.png')
    plt.show()

    ##################################################
    ##################################################
    ##################################################
    # Seguidor de Referencia
    # Calculate Nx and Nu
    # Calculate Nx and Nu
    M = np.block([
        [A, B],
        [C, D]
    ])
    
    rhs = np.zeros((A.shape[0] + C.shape[0], 1))
    rhs[-1] = 1

    # Resolver para Nx e Nu
    sol = np.linalg.solve(M, rhs)
    Nx = sol[:A.shape[1]]
    Nu = sol[A.shape[1]:]

    A_aug = np.block([
        [A, B @ Nu],
        [-C, np.zeros((1, 1))]
    ])
    
    B_aug = np.block([
        [B],
        [0]
    ])

    # Polos do Observador
    # p1o = -9.5 - 45j
    # p2o = np.conj(p1o)
    # p3o = -9 - 48j
    # p4o = np.conj(p3o)
    # p5o = -11 - 35j
    # p6o = np.conj(p5o)
    # p7o = -12 
    # pobs = [p1o, p2o, p3o, p4o, p5o, p6o, p7o]

    p1o = -4.5 - 45j
    p2o = np.conj(p1o)
    p3o = -4 - 48j
    p4o = np.conj(p3o)
    p5o = -6 - 35j
    p6o = np.conj(p5o)
    pobs = [p1o, p2o, p3o, p4o, p5o, p6o]
    print('pobs', pobs)

    # Polos do Controlador
    # p1c = -3 + 26j
    # p2c = np.conj(p1c)
    # p3c = -4 + 24j
    # p4c = np.conj(p3c)
    # p5c = -5 + 22j
    # p6c = np.conj(p5c)
    # p7c = -6  
    # pctrl = [p1c, p2c, p3c, p4c, p5c, p6c, p7c]

    p1c = -0.04 + 20j
    p2c = np.conj(p1c)
    p3c = -0.05 + 22j
    p4c = np.conj(p3c)
    p5c = -0.06 + 24j
    p6c = np.conj(p5c)
    p7c = -0.06  
    pctrl = [p1c, p2c, p3c, p4c, p5c, p6c, p7c]


    L_aug = signal.place_poles(A_aug.T, np.vstack((C.T, np.zeros((1, 1)))), pobs).gain_matrix.T

    K_aug = signal.place_poles(A_aug, B_aug, pctrl).gain_matrix

    print("Dimensions of L_aug:", L_aug.shape)
    print("Dimensions of C:", C.shape)
    print("Dimensions of B_aug:", B_aug.shape)
    print("Dimensions of K_aug:", K_aug.shape)

    A_aug_obs = np.block([
        [A_aug - B_aug @ K_aug, B_aug @ K_aug],
        [L_aug @ np.vstack((C, np.zeros((1, C.shape[1])))), A_aug - L_aug @ np.vstack((C, np.zeros((1, C.shape[1])))) - B_aug @ K_aug]
    ])
    
    B_aug_obs = np.block([
        [B_aug],
        [np.zeros_like(B_aug)]
    ])
    
    C_aug_obs = np.block([C, np.zeros((C.shape[0], C.shape[1]))])
    D_aug_obs = D

    sys_aug_obs = signal.StateSpace(A_aug_obs, B_aug_obs, C_aug_obs, D_aug_obs)

    t0 = 0
    dt = 0.001
    tf = 100
    t = np.arange(t0, tf, dt)

    r = np.ones(len(t)) * 10  

    x0 = np.zeros(A_aug_obs.A.shape[0])
    x0[2] = 0.1  


    _, y, x_aug = signal.lsim(sys_aug_obs, U=r, T=t, X0=x0)

    x_real = x_aug[:, :7] 
    x_est = x_aug[:, 7:] 

    error_x1 = x_real[:, 0] - x_est[:, 0]
    error_x2 = x_real[:, 1] - x_est[:, 1]
    error_psi = x_real[:, 2] - x_est[:, 2]


    plt.figure()
    plt.plot(t, y, label='System Output $\psi$')
    plt.plot(t, r, '--', label='Reference $\psi_{ref}$')
    plt.title('System Output and Reference Tracking')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle $\psi$ (degrees)')
    plt.legend()
    plt.grid()
    plt.show()


    plt.figure()
    plt.plot(t, error_x1, label='Error in $x1$')
    plt.title('Observation Error for $x1$')
    plt.xlabel('Time (s)')
    plt.ylabel('Error in $x1$')
    plt.legend()
    plt.grid()
    plt.show()


    # plt.figure()
    # plt.plot(t, error_x2, label='Error in $x2$')
    # plt.title('Observation Error for $x2$')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Error in $x2$')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.plot(t, error_psi, label='Error in $\psi$')
    # plt.title('Observation Error for $\psi$')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Error in $\psi$')
    # plt.legend()
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()
