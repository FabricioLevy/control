import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
from functions import ControlFunctions
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
    print("Polos do Sistema em Malha Aberta:", poles)
    print("Zeros do Sistema em Malha Aberta:", zeros)

    # quit()

    wn_values = np.arange(10, 16, 1)
    s = ctrl.TransferFunction.s
    itae_sys = itae_pid_tuning(sys_tf, wn_values)

    plt.figure()
    for wn, itae in zip(wn_values, itae_sys):
        pid_controller = itae / (s**2 + 2*zeta*wn*s + wn**2)
        sys_closed_loop = ctrl.feedback(pid_controller * sys_tf, 1)
        t = np.linspace(0, 6, 1000)
        t, y = ctrl.step_response(sys_closed_loop, T=t)
        plt.plot(t, y, label=f'wn={wn} rad/s')

    plt.title('Sintonia ITAE para diferentes valores de wn')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ângulo psi (rad)')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT + 'resposta_degrau_itae_wn.png')
    plt.show()
    # quit()

    plot_bode(sys_closed_loop, OUTPUT)
    plot_rlocus(sys_tf, OUTPUT)

    Kp_values = np.linspace(0, 10000, 500)
    Ki_values = np.linspace(0, 10000, 500)
    Kd_values = np.linspace(0, 10000, 500)

    plot_rlocus_proportional(sys_tf, Kp_values, OUTPUT)
    plot_rlocus_integral(sys_tf, Ki_values, OUTPUT)
    plot_rlocus_derivative(sys_tf, Kd_values, OUTPUT)

if __name__ == "__main__":
    main()
