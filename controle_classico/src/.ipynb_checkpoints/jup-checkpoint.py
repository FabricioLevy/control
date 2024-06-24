import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from ipywidgets import interactive, FloatSlider

def interactive_root_locus(real1=-10, imag1=10):
    # Define as matrizes do sistema (Exemplo)
    A = np.array([[0, 1], [-2, -3]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])

    # Define um par de polos conjugados para controle
    poles = [complex(real1, imag1), complex(real1, -imag1)]

    # Calcula o controlador com os polos especificados
    K = ctrl.place(A, B, poles)
    A_new = A - B.dot(K)
    sys = ctrl.ss(A_new, B, C, D)
    
    # Plota o lugar das raízes
    plt.figure(figsize=(12, 6))
    ctrl.root_locus(sys)
    plt.title(f'Root Locus for Poles: {poles}')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.show()
    
    # Plota a resposta ao degrau
    T, Y = ctrl.step_response(sys)
    plt.figure(figsize=(12, 6))
    plt.plot(T, Y)
    plt.title('Step Response')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Response')
    plt.grid(True)
    plt.show()

# Widgets interativos para ajustar os polos
interactive_plot = interactive(interactive_root_locus, 
                               real1=FloatSlider(value=-10, min=-20, max=0, step=0.5),
                               imag1=FloatSlider(value=10, min=0, max=20, step=0.5))
interactive_plot
