import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import os
import pandas as pd
from functions import ControlFunctions


def generate_poles(real_start, real_end, real_step, imag_start, imag_end, imag_step):
    real_parts = np.arange(real_start, real_end, real_step)
    imag_parts = np.arange(imag_start, imag_end, imag_step)
    poles = []
    for real in real_parts:
        for imag in imag_parts:
            poles.append(complex(real, imag))
            poles.append(complex(real, -imag))  # Adiciona o conjugado
    return poles

def test_poles(poles, A, B, C, D):
    responses = []
    for pole_set in poles:
        try:
            K = place(A, B, pole_set)
            A_new = A - B @ K
            sys = ss(A_new, B, C, D)
            T, Y = step_response(sys)
            responses.append((pole_set, T, Y))
        except Exception as e:
            print(f"Failed for poles {pole_set}: {str(e)}")
    return responses

# Exemplo de uso:
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Gere uma lista de polos
poles = generate_poles(-10, 0, 1, 0, 10, 1)

# Teste os polos
responses = test_poles([poles], A, B, C, D)

# Plote as respostas para avaliação
for pole_set, T, Y in responses:
    plt.figure()
    plt.plot(T, Y, label=f'Poles: {pole_set}')
    plt.title(f'Response for Poles {pole_set}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Response')
    plt.legend()
    plt.grid(True)
    plt.show()
