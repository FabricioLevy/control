import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import sys, os


class ControlFunctions:

    
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

    def criar_matriz_A(self, m, k, c, m0, Mt, Kt, Ct):
        
        n = len(m)
        A = np.zeros((12, 12))
        A[:6, 6:12] = np.eye(6)
        
        for i in range(n):
            A[6+i, 6+i] = -c[i] / m[i]
            A[6+i, 6+i+4] = c[i] / m[i]
            A[6+i, i*4] = -k[i] / m[i]
            A[6+i, i*4+4] = k[i] / m[i]
        
        a111 = -(c[0] + c[1]) / (Mt + m0)
        a1112 = Ct / (Mt + m0)
        a112 = k[0] / (Mt + m0)
        a115 = -(Kt + k[0] + k[1]) / (Mt + m0)
        a116 = Kt / (Mt + m0)
        a117 = c[0] / (Mt + m0)
        a118 = c[1] / (Mt + m0)
        a1210 = c[1] / (Mt + m0)
        a1211 = -(Ct + c[0] + c[1]) / (Mt + m0)
        a123 = k[0] / (Mt + m0)
        a124 = k[1] / (Mt + m0)
        a125 = Kt / (Mt + m0)
        a126 = -(Kt + k[0] + k[1]) / (Mt + m0)
        a129 = c[0] / (Mt + m0)
        a1212 = (Ct + c[0] + c[1]) / (Mt + m0)

        A[-1, [0, 1, 4, 5, 6, 7, 9, 10, 11]] = [a111, a112, a115, a116, a117, a118, a1210, a1211, a1212]
        
        return A

    def massa_i(self, m_d, i, hf, l):
        
        termo = (2 * i + 1) * np.pi * hf / l
        m_i = 8 * np.tanh(termo) / (np.pi**3 * hf * (2 * i + 1)**3 / l) * m_d
        return m_i

    def rigidez_i(self, hf, m_d, g, l, i):
        
        termo = (2 * i + 1) * np.pi * hf / l
        ki = (8 * m_d * g * (np.tanh(termo)**2)) / (hf * (2 * i + 1)**2)
        return ki

    def amortecimento_i(self, m_i, fn, df):
        
        ci = 2 * m_i * fn * df
        return ci

    def massa_0(self, m, m_d):
        
        m_0 = m_d - np.sum(m)
        return m_0
    
