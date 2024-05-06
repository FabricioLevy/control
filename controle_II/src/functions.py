import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import sys, os
import scipy.linalg as la
from control import ctrb, obsv
from scipy.special import gamma
from scipy.linalg import fractional_matrix_power


class ControlFunctions:

    
    def massa_0(self, m, m_d):
        m_0 = m_d
        for i in range(len(m)):
            m_0 -= m[i]
        return m_0

    def massa_i(self, m_d, i, hf, l):
        return 8 * np.tanh(np.pi * hf * (2 * i + 1) / l) / (np.pi**3 * hf * (2 * i + 1)**3 / l) * m_d

    def amortecimento_i(self, m_i, fn, df):
        return 2 * m_i * fn * df

    def rigidez_i(self, hf, m_d, g, l, i):
        return (8 * m_d * g * (np.tanh((2 * i + 1) * np.pi * hf / l))**2) / (hf * (2 * i + 1)**2)
 
    def Mtransicao(self, A, dt, n):
        soma = np.zeros_like(A)
        for i in range(n+1):
            termo = np.linalg.matrix_power(A*dt, i) / gamma(i+1)
            soma += termo
        return soma

    def OMEGA(self, A, B, u, dt, n, x0):
        soma2 = np.zeros_like(A)
        soma1 = self.Mtransicao(A, dt, n)
        for i in range(n):
            termo = np.linalg.matrix_power(A*dt, i) / gamma(i+2)
            soma2 += termo
        x = np.dot(soma1, x0) + np.dot(soma2, B) * dt * u
        return x
