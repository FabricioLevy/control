import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
from functions import ControlFunctions
import matplotlib.ticker as mticker
import pandas as pd


def find_ku_and_tu(sys_tf, start=0.1, end=100, step=0.1):
    Ku = start

    found = False
    previous_output = None
    t_final = 50
    
    while not found and Ku <= end:
        Kp = Ku
        sys_cl = ctrl.feedback(Kp * sys_tf)
        t, yout = ctrl.step_response(sys_cl, T=np.linspace(0, t_final, 50))
        
        # Diagnóstico da resposta
        plt.plot(t, yout, label=f'Kp={Kp:.1f}')
        plt.title('Resposta do Sistema para Diferentes Kp')
        plt.xlabel('Tempo')
        plt.ylabel('Resposta')
        plt.legend()
        plt.grid(True)
        
        if previous_output is not None and np.any(yout * previous_output < 0):
            found = True
        previous_output = yout

        Ku += step

    plt.show()

    if found:
        t, yout = ctrl.step_response(sys_cl, T=np.linspace(0, t_final, 500))
        peaks = np.where(np.diff(np.sign(np.diff(yout))) == -2)[0] + 1
        if len(peaks) > 1:
            Tu = np.mean(np.diff(t[peaks]))
            return Ku - step, Tu  # Retornar o Ku encontrado no passo anterior
    return None, None


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
    # print(Mt)
    # quit()
    hcm_t = 1
    hcm_0 = 1 + h0
    hcm_1 = 1 + h1
    hcm_2 = 1 + h2
    hcm_3 = 1 + h3 ############# adicionar m3

    Hcm = (hcm_t*M_vazio + hcm_0*m0 + hcm_1*m1 + hcm_2*m2 + hcm_3*m3)/Mt
    print('Hcm', Hcm)
    Fc = (Mt)*aceleracao_centripeta
    print('Fc', Fc)
    My = 2*Hcm*Fc
    print('My', My)

    sys_ss = ctrl.ss(A, B, C, D)
    sys_tf = ctrl.ss2tf(sys_ss)
    print("Função de Transferência do Sistema:")

    # Plotar e imprimir o local das raízes do sistema
    rlist, klist = ctrl.root_locus(sys_tf, xlim=(-4, 1), ylim=(-6, 6), plot=False)
    print("Local das Raízes do Sistema:")
    # plt.savefig(OUTPUT + 'kp_root_locus')
    # plt.show()
    # for r, k in zip(rlist, klist):
    #     print(f"Valor de k: {k}, Raízes: {r}")

    kp = 3.377738e+07
    sys_tf_num = sys_tf.num[0][0]
    sys_tf_den = sys_tf.den[0][0]

    max_len = max(len(sys_tf_num), len(sys_tf_den))

    sys_tf_num_padded = np.pad(sys_tf_num, (max_len - len(sys_tf_num), 0))
    sys_tf_den_padded = np.pad(sys_tf_den, (max_len - len(sys_tf_den), 0))
    
    den_new = sys_tf_den_padded + kp * sys_tf_num_padded
    den_poly = np.poly1d(den_new)

    tf = ctrl.TransferFunction(sys_tf_num_padded, den_poly.coeffs)

    rlist, klist = ctrl.root_locus(tf, plot=False)
    ki = 6.167e7
    # plt.savefig(OUTPUT + 'ki_root_locus.png')
    # print(rlist, klist)

    den_new_1 = sys_tf_den_padded + kp * sys_tf_num_padded + ki*np.pad([1], (max_len -1, 0))

    den_poly = np.poly1d(den_new_1)

    tf_new = ctrl.TransferFunction(sys_tf_num_padded, den_poly.coeffs)

    rlist, klist = ctrl.root_locus(tf_new, plot=False)

    kd = 9.798e8
    # plt.show()

    num_pid = [kd, kp, ki]
    den_pid = [1, 0]

    # pid_tf = ctrl.TransferFunction(num_pid, den_pid)

    controle_tf = ctrl.TransferFunction(num_pid, den_pid)


    poles = [(-4+20j), (-4-20j), 
           (-5+22.5j), (-5-22.5j), 
           (-6+24j), (-6-24j)]

    K = ctrl.place(A, B, poles)
    A_new = A - B.dot(K)
    sys = ctrl.ss(A_new, B, C, D)

    G = ctrl.ss2tf(sys)
    G_scaled = G * My

    # sys_ss = ctrl.ss(A, B, C, D)
    # Gp = ctrl.ss2tf(sys_ss)
    # G_f = Gp * My

    system_tf = ctrl.series(controle_tf, G_scaled)
    tf_closed = ctrl.feedback(system_tf, 1)
    # print(tf_closed)

    # loop_tf = controle_tf*ctrl.ss2tf(sys_ss)

    t, y = ctrl.step_response(G_scaled)
    y = y*(180/np.pi)
    plt.figure(figsize=(12, 6))
    plt.xlabel('Tempo')
    plt.ylabel('Resposta do Sistema')
    plt.title(f'Resposta do Sistema Controlado - PID')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
