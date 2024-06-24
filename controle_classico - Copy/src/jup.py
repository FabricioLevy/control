import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from tkinter import Tk, Label, DoubleVar, Scale, HORIZONTAL

def update_plot(real, imag):
    poles = [complex(real, imag), complex(real, -imag)]
    K = ctrl.place(A, B, poles)
    A_new = A - B.dot(K)
    sys = ctrl.ss(A_new, B, C, D)

    plt.figure()
    ctrl.root_locus(sys, Plot=True)
    plt.title(f'Root Locus for Poles: {poles}')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.show()

def on_value_change(val):
    real_val = real_slider.get()
    imag_val = imag_slider.get()
    update_plot(real_val, imag_val)

A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

root = Tk()
root.title("Pole Adjuster")

real_slider = Scale(root, from_=-20, to=0, resolution=0.1, orient=HORIZONTAL, label="Real Part", command=on_value_change)
real_slider.pack()

imag_slider = Scale(root, from_=0, to=30, resolution=0.1, orient=HORIZONTAL, label="Imaginary Part", command=on_value_change)
imag_slider.pack()

root.mainloop()
