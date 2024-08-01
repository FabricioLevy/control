import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
from functions import ControlFunctions
from control.matlab import bode
from control import ctrb, obsv
import matplotlib.ticker as mticker

def itae_pid_tuning(sys_tf, wn_values):
    """
    Calcula os ganhos do controlador PID utilizando fórmulas de sintonia ITAE para sistemas de ordem superior.
    para diferentes valores de wn.
    """
    s = ctrl.TransferFunction.s
    pid_params = []

    for wn in wn_values:
        itae_poly = s**7 + 2.217*wn*s**6 + 6.745*wn**2*s**5 + 9.349*wn**3*s**4 + 11.58*wn**4*s**3 + 8.68*wn**5*s**2 + wn**7
        itae_sys = itae_poly / wn**7
        pid_params.append(itae_sys)

    return pid_params

def format_axes(ax):
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

def plot_bode(sys_closed_loop, OUTPUT):
    w, mag, phase = signal.bode((sys_closed_loop.num[0][0], sys_closed_loop.den[0][0]))

    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    plt.semilogx(w, mag)
    plt.title('Bode Plot')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="--")
    format_axes(ax1)

    ax2 = plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Phase (degrees)')
    plt.grid(True, which="both", ls="--")
    format_axes(ax2)

    plt.savefig(OUTPUT + 'bode_plot_pid_itae.png')
    plt.show()

def plot_rlocus(sys_tf, OUTPUT):
    plt.figure()
    ctrl.rlocus(sys_tf)
    plt.title('Lugar das Raízes')
    plt.xlabel('Parte Real (seconds^{-1})')
    plt.ylabel('Parte Imaginária (seconds^{-1})')
    plt.grid(True)
    ax = plt.gca()
    format_axes(ax)
    plt.savefig(OUTPUT + 'lugar_das_raizes.png')
    plt.show()

def plot_rlocus_proportional(sys_tf, Kp_values, OUTPUT):
    plt.figure()
    ctrl.rlocus(sys_tf * ctrl.TransferFunction([1], [1]), kvect=Kp_values)
    plt.title('Lugar das raízes para o ganho proporcional')
    plt.xlabel('Eixo real (seconds^{-1})')
    plt.ylabel('Eixo imaginário (seconds^{-1})')
    plt.grid(True)
    ax = plt.gca()
    format_axes(ax)
    plt.savefig(OUTPUT + 'lugar_das_raizes_proporcional.png')
    plt.show()

def plot_rlocus_integral(sys_tf, Ki_values, OUTPUT):
    plt.figure()
    ctrl.rlocus(sys_tf * ctrl.TransferFunction([1, 0], [1, 0]), kvect=Ki_values)
    plt.title('Lugar das raízes para o ganho integrativo')
    plt.xlabel('Eixo real (seconds^{-1})')
    plt.ylabel('Eixo imaginário (seconds^{-1})')
    plt.grid(True)
    ax = plt.gca()
    format_axes(ax)
    plt.savefig(OUTPUT + 'lugar_das_raizes_integral.png')
    plt.show()

def plot_rlocus_derivative(sys_tf, Kd_values, OUTPUT):
    plt.figure()
    ctrl.rlocus(sys_tf * ctrl.TransferFunction([1, 0], [1]), kvect=Kd_values)
    plt.title('Lugar das raízes para o ganho derivativo')
    plt.xlabel('Eixo real (seconds^{-1})')
    plt.ylabel('Eixo imaginário (seconds^{-1})')
    plt.grid(True)
    ax = plt.gca()
    format_axes(ax)
    plt.savefig(OUTPUT + 'lugar_das_raizes_derivativo.png')
    plt.show()

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
    print('mf', mf)
    # quit()

    # m0 = fx.massa_i(mf, 0, hf, L)
    m1 = fx.massa_i(mf, 1, hf, L)
    m2 = fx.massa_i(mf, 2, hf, L)
    m3 = fx.massa_i(mf, 3, hf, L)
    m0 = mf - m1 - m2 - m3
    print('m0', m0)
    print('m1', m1)
    print('m2', m2)
    print('m3', m3)

    k0 = fx.calcular_ki(hf, L, g, mf, 0)
    k1 = fx.calcular_ki(hf, L, g, mf, 1)
    k2 = fx.calcular_ki(hf, L, g, mf, 2)

    print('k0', k0)
    print('k1', k1)
    print('k2', k2)


    h0 = fx.calcular_hi(L, hf, 0)
    h1 = fx.calcular_hi(L, hf, 1)
    h2 = fx.calcular_hi(L, hf, 2)
    h3 = fx.calcular_hi(L, hf, 3)

    print('h0', h0)
    print('h1', h1)
    print('h2', h2)
    print('h3', h3)

    I0 = fx.momento_inercia(M_vazio, L, H)
    print('I0', I0)
    If = I0 + m0 * h0**2 + m1 * h1**2 + m2 * h2**2 + m3 * h3**2
    print('If', If)

    omega1 = np.sqrt(k1 / m1)
    omega2 = np.sqrt(k2 / m2)
    
    print('omega1', omega1)
    print('omega2', omega2)

    zeta = ((6 * 10**(-6)) / (np.sqrt(g) * L**(3 / 2)))**(1 / 2)
    
    print('zeta', zeta)
    # quit()


    c1 = fx.calcular_ci(m1, omega1, zeta)
    c2 = fx.calcular_ci(m2, omega2, zeta)

    print('c1', c1)
    print('c2', c2)

    k = 15000
    cd = 120
    # quit()
    
    print('k', k)
    print('cd', cd)

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
    print(sys_ss)
    # quit()
    sys_tf = ctrl.ss2tf(sys_ss)
    print("Função de Transferência do Sistema:")
    print(sys_tf)
    # quit()

    poles = ctrl.poles(sys_tf)
    zeros = ctrl.zeros(sys_tf)

    print("Polos:", poles)
    print("Zeros:", zeros)
    quit()



    num = sys_tf.num[0][0]
    den = sys_tf.den[0][0]

    num[0] = 0
    num[1] = 0
    num[2] = 0

    # den[0] = 0
    # den[1] = 0
    # den[2] = 0

    print('num', num)
    print('den', den)
    # quit()

    sys_tf_adjusted = ctrl.TransferFunction(num, den)
    print("Função de Transferência Ajustada do Sistema:")
    print(sys_tf_adjusted)
    # quit()
    fx.plot_poles(sys_tf_adjusted, OUTPUT + 'polos_zeros_malha_aberta.png')

    char_poly = np.poly(A)
    print("Polinômio Característico do Sistema:", char_poly)

    # Construindo a tabela de Routh
    routh_array = fx.routh_hurwitz(char_poly)
    print("Tabela de Routh-Hurwitz:")

    # Verificando estabilidade
    stable = np.all(routh_array[:, 0] > 0)
    print(stable)


    #################### Malha Fechada #############
    velocidade_kmh = 80  # km/h
    velocidade_ms = velocidade_kmh / 3.6  # m/s

    raio = 300  # m

    aceleracao_centripeta = (velocidade_ms ** 2) / raio

    Mt = M_vazio + m0 + m1 + m2 + m3
    print('Mt', Mt)
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

    step_amplitude = My

    G_scaled = sys_tf * step_amplitude
    t, y = ctrl.step_response(G_scaled)

    y_degress = y*(180/np.pi)

    plt.figure()
    plt.plot(t, y_degress)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (graus)')
    # plt.ylim(0, 90)
    # plt.xlim(0, 1)
    plt.title('Resposta com Amplitude')
    plt.grid()
    plt.savefig(OUTPUT + 'resposta_em_malha_aberta.png')
    plt.show()

    plt.figure()
    plt.plot(t, y_degress)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo (graus)')
    plt.ylim(0, 90)
    plt.xlim(0, 1)
    plt.title('Resposta com Amplitude com Limite de Tombamento')
    plt.grid()
    plt.savefig(OUTPUT + 'resposta_em_malha_aberta.png')
    plt.show()
    # quit()

    # Bode
    plt.figure()
    w = np.logspace(-1.5,1,200)
    mag, phase, freq = bode(sys_tf, w, Hz=True, dB=True)
    plt.tight_layout()
    plt.savefig(OUTPUT + 'diagrama_bode.png')
    plt.show()

    # Analise de Controlabilidade e Observabilidade
    ctrl_matrix = ctrb(A, B)
    obsv_matrix = obsv(A, C)

    controlable = np.linalg.matrix_rank(ctrl_matrix) == A.shape[0]
    observable = np.linalg.matrix_rank(obsv_matrix) == A.shape[0]

    print("O sistema é controlável?", controlable)
    print("O sistema é observável?", observable)

    polos_list = [
        # [-0.05 - 26j, -0.05 + 26j, -0.06 - 24j, -0.06 + 24j, -0.06 - 22j, -0.06 + 22j],
        [-0.01 - 10j, -0.01 + 10j, -0.025 - 40j, -0.025 + 40j, -0.035 - 60j, -0.035 + 60j],
        # [-1 - 30j, -1 + 30j, -2 - 28j, -2 + 28j, -3 - 26j, -3 + 26j],
        # [-2 - 28j, -2 + 28j, -3 - 26j, -3 + 26j, -4 - 24j, -4 + 24j],
        # [-3 - 26j, -3 + 26j, -4 - 24j, -4 + 24j, -5 - 22j, -5 + 22j], ######
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
        num_f = Gf.num[0][0]
        den_f = Gf.den[0][0]

        num_f[0] = 0
        num_f[1] = 0
        num_f[2] = 0

        den_f[0] = 0
        den_f[1] = 0
        den_f[2] = 0
        # print(Gf)
        # quit()

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

        fx.plot_poles_mult(sys_ss, sys_closed, OUTPUT + 'polos_zeros_comp.png')
        plt.show()




if __name__ == "__main__":
    main()
