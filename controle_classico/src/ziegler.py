import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import control as ctrl
from functions import ControlFunctions
import matplotlib.ticker as mticker


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

    sys_ss = ctrl.ss(A, B, C, D)
    sys_tf = ctrl.ss2tf(sys_ss)
    print("Função de Transferência do Sistema:")
    print(sys_tf)

    sys_tf_num = sys_tf.num[0][0]
    sys_tf_den = sys_tf.den[0][0]

    sys_tf_num[np.abs(sys_tf_num) < 1e-10] = 0
    sys_tf_den[np.abs(sys_tf_den) < 1e-10] = 0

    sys_tf = ctrl.TransferFunction(sys_tf_num, sys_tf_den)

    print("Função de Transferência Simplificada:")
    print(sys_tf)

    # Verificar a estabilidade
    poles = np.array(ctrl.poles(sys_tf))
    if np.any(np.real(poles) > 0):
        print("Sistema instável. Polos:", poles)
    else:
        print("Sistema estável. Polos:", poles)

    # Define Ku e Tu manualmente baseado na análise gráfica
    Ku = 2.0  # Ajuste este valor com base na análise das oscilações sustentadas
    Tu = 10  # Ajuste este valor para refletir o período de oscilação

    print(f"Ganho crítico (Ku): {Ku}")
    print(f"Período crítico (Tu): {Tu}")

    # Escolha o tipo de controlador: 'P', 'PI' ou 'PID'
    controlador_tipo = 'PID'  # Altere conforme necessário

    if controlador_tipo == 'P':
        Kp = 0.5 * Ku
        Ti = float('inf')
        Td = 0

    elif controlador_tipo == 'PI':
        Kp = 0.45 * Ku
        Ti = Tu / 1.2
        Td = 0

    elif controlador_tipo == 'PID':
        Kp = 0.60 * Ku
        Ti = 0.50 * Tu
        Td = 0.125 * Tu

    # Criar controlador baseado na escolha
    if controlador_tipo == 'P':
        controlador = ctrl.TransferFunction([Kp], [1])
    elif controlador_tipo == 'PI':
        controlador = ctrl.TransferFunction([Kp * Ti, Kp], [Ti, 0])
    elif controlador_tipo == 'PID':
        controlador = ctrl.TransferFunction([Kp * Td * Ti, Kp * Ti, Kp], [Ti, 0, 0])
    
    # Sistema em malha fechada com o controlador escolhido
    sys_cl = ctrl.feedback(controlador * sys_tf)

    # Resposta ao degrau
    t, yout = ctrl.step_response(sys_cl)

    # Plotar a resposta
    plt.plot(t, yout)
    plt.xlabel('Tempo')
    plt.ylabel('Resposta do Sistema')
    plt.title(f'Resposta do Sistema Controlado ({controlador_tipo})')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
