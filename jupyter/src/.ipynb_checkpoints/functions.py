import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import sys, os
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import minimize
import scipy.signal as signal



class ControlFunctions:

    def lqr(self, A, B, Q, R):

        P = np.matrix(ctrl.care(A, B, Q, R)[0])
    
        K = np.matrix(ctrl.acker(A, B, P))
        return K, P
    
    def plot_poles_lqr(self, sys_open, sys_closed, output_path):

        poles_open = ctrl.poles(sys_open)
        zeros_open = ctrl.zeros(sys_open)
        poles_closed = ctrl.poles(sys_closed)
        zeros_closed = ctrl.zeros(sys_closed)

        plt.figure()
        
      
        plt.scatter(np.real(poles_open), np.imag(poles_open), marker='x', color='red', label='Polos (Malha Aberta)')
        plt.scatter(np.real(zeros_open), np.imag(zeros_open), marker='o', color='blue', label='Zeros (Malha Aberta)')
        
    
        plt.scatter(np.real(poles_closed), np.imag(poles_closed), marker='x', color='green', label='Polos (Malha Fechada)')
        plt.scatter(np.real(zeros_closed), np.imag(zeros_closed), marker='o', color='purple', label='Zeros (Malha Fechada)')
        
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        plt.xlim(-0.05, 0.05)  
        plt.ylim(np.min(np.imag(poles_open + poles_closed)) - 20, np.max(np.imag(poles_open + poles_closed)) + 20)  # Ajuste dos limites do eixo y conforme necessário
        plt.title('Polos e zeros das malhas aberta e fechada')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.grid()
        plt.legend()
        plt.savefig(output_path)
        # plt.show()


    def matriz_transicao_estados(self, A, delta_t):

        Phi = expm(A * delta_t)
        return Phi

    
    def integrando(self, A, delta_t, tau):

        return self.matriz_transicao_estados(A, delta_t - tau)
    
    def matriz_delta(self, A, delta_t):

        n = A.shape[0]
        Gamma = np.zeros((n, n))
        
        # Integrando para cada elemento da matriz
        for i in range(n):
            for j in range(n):
                integrando_func = lambda tau: self.integrando(A, delta_t, tau)[i, j]
                Gamma[i, j], _ = quad(integrando_func, 0, delta_t)
        
        return Gamma


    def calcular_delta_t(self, A, factor=0.1):

        # Calcula os autovalores da matriz A
        eigenvalues = np.linalg.eigvals(A)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        
        # Define Δt como uma fração do inverso do maior autovalor
        delta_t = factor / max_eigenvalue
        return delta_t

    def matriz_transicao_estados(self, A, delta_t):

        # Calcula a matriz de transição de estados
        Phi = expm(A * delta_t)
        return Phi



    def open_loop_response(self, amplitude):
        # Simulação de dados
        t = np.linspace(0, 10, 100)  # tempo de 0 a 10 segundos
        response = amplitude * (1 - np.exp(-t))  # resposta hipotética
        return t, response

    def plot_poles(self, sys, output_path):

        poles = ctrl.poles(sys)
        zeros = ctrl.zeros(sys)

        plt.figure()
        plt.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Polos')
        plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros')
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        plt.xlim(-0.05, 0.05)  
        plt.ylim(np.min(np.imag(poles)) - 20, np.max(np.imag(poles)) + 20)  # Ajuste dos limites do eixo y conforme necessário
        plt.title('Polos e zeros da malha aberta')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.grid()
        plt.legend()
        plt.savefig(output_path)
        # plt.show()

    def plot_poles_mult(self, sys_open, sys_closed, output_path):

        poles_open = ctrl.poles(sys_open)
        zeros_open = ctrl.zeros(sys_open)
        poles_closed = ctrl.poles(sys_closed)
        zeros_closed = ctrl.zeros(sys_closed)

        plt.figure()
        
        # Plot open-loop poles and zeros
        plt.scatter(np.real(poles_open), np.imag(poles_open), marker='x', color='red', label='Polos (Malha Aberta)')
        plt.scatter(np.real(zeros_open), np.imag(zeros_open), marker='o', color='blue', label='Zeros (Malha Aberta)')
        
        # Plot closed-loop poles and zeros
        plt.scatter(np.real(poles_closed), np.imag(poles_closed), marker='x', color='green', label='Polos (Malha Fechada)')
        plt.scatter(np.real(zeros_closed), np.imag(zeros_closed), marker='o', color='purple', label='Zeros (Malha Fechada)')
        
        plt.axhline(0, color='black', lw=1)
        plt.axvline(0, color='black', lw=1)
        # plt.xlim(-0.05, 0.05)  
        # plt.ylim(np.min(np.imag(poles_open + poles_closed)) - 20, np.max(np.imag(poles_open + poles_closed)) + 20)  # Ajuste dos limites do eixo y conforme necessário
        plt.title('Polos e zeros das malhas aberta e fechada')
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.grid()
        plt.legend()
        # plt.show()
        plt.savefig(output_path)

    def plot_comparative_poles(self, A, B, C, D, pobs, pctrl, output_path):

        ko = signal.place_poles(A.T, C.T, pobs).gain_matrix.T
        ko = ko.reshape(-1, 1) 
        O = A - np.dot(ko, C)

        kc = signal.place_poles(A, B, pctrl).gain_matrix
        Ac = A - np.dot(B, kc)

        sys_obs = signal.StateSpace(O, B, C, D)
        sys_ctrl = signal.StateSpace(Ac, B, C, D)

        num_obs, den_obs = signal.ss2tf(O, B, C, D)
        num_ctrl, den_ctrl = signal.ss2tf(Ac, B, C, D)

        zeros_obs, poles_obs, _ = signal.tf2zpk(num_obs, den_obs)
        zeros_ctrl, poles_ctrl, _ = signal.tf2zpk(num_ctrl, den_ctrl)

        # Plot poles and zeros
        plt.figure(figsize=(10, 8))
        plt.scatter(np.real(poles_obs), np.imag(poles_obs), marker='o', color='red', label='Polos Observador')
        plt.scatter(np.real(poles_ctrl), np.imag(poles_ctrl), marker='x', color='blue', label='Polos Controlador')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title('Comparativo Entre Polos do Observador e Controlador')
        plt.legend()
        plt.grid()
        plt.savefig(output_path)
        # plt.show()

    def routh_hurwitz(self, coeffs):
        degree = len(coeffs) - 1
        routh_array = np.zeros((degree + 1, int(np.ceil((degree + 1) / 2))))
        
        routh_array[0, :len(coeffs[::2])] = coeffs[::2]
        routh_array[1, :len(coeffs[1::2])] = coeffs[1::2]
        
        for i in range(2, degree + 1):
            for j in range(0, routh_array.shape[1] - 1):
                routh_array[i, j] = ((routh_array[i-1, 0] * routh_array[i-2, j+1] - routh_array[i-2, 0] * routh_array[i-1, j+1]) /
                                    routh_array[i-1, 0])
        
        return routh_array

    def calcular_ci(self, m_i, w_i, zeta, i):
        ci = 2 * m_i * w_i * zeta
        return ci

    def calcular_ki(self, h_f, l, g, m_f, i):
        numerador = 8 * np.tanh(((2 * i) + 1) * np.pi * h_f / l)**2
        denominador = (2 * i + 1)
        ki = (m_f * g / h_f) * (numerador / denominador)
        return ki

    def calcular_hi(self, L, h_f, i):
        numerador = np.tanh(((2*i) + 1)*np.pi*(h_f/(2*L)))
        denominador = (2*i+1)*np.pi*(h_f/(2*L))
        h_i = (0.5 - (numerador/denominador))

        return h_i

    def momento_inercia(self, massa, largura, altura):
        I0 = (1/12)*massa*(largura**2 + altura**2)
        return I0

    def calcular_parametros(self, m_d, hf, l, g, n, Mt, Kt, Ct, visc_cin):
        
        stigma = np.sqrt(visc_cin / (l**(3/2) * np.sqrt(g)))
        m = np.zeros(n)
        k = np.zeros(n)
        omega_n = np.zeros(n)
        c = np.zeros(n)
        
        for i in range(1, n+1):  # Começa em 1 e vai até n inclusivo
            m[i-1] = self.massa_i(m_d, i, hf, l)  # i-1 para acessar o índice correto do array
            k[i-1] = self.rigidez_i(hf, m_d, g, l, i)
            omega_n[i-1] = np.sqrt(k[i-1] / m[i-1])
            c[i-1] = self.amortecimento_i(m[i-1], omega_n[i-1], stigma)

        m0 = self.massa_0(m, m_d)
        return m, k, c, m0


    def massa_i(self, m_d, i, hf, l):
        
        termo = (2 * i + 1) * np.pi * hf / l
        m_i = 8 * np.tanh(termo) / (np.pi**3 * hf * (2 * i + 1)**3 / l) * m_d
        return m_i

    def amortecimento_i(self, m_i, fn, df):
        
        ci = 2 * m_i * fn * df
        return ci

    def massa_0(self, m, m_d):
        
        m_0 = m_d - np.sum(m)
        return m_0
    
