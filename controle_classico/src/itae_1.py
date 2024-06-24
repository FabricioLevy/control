import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Parâmetros do sistema
Ta = 2
zeta = 0.5
omega = 4 / (Ta * zeta)

# Função de transferência do processo
nump = [1]
denp = [0.26, 1.26, 1]
Gp = ctrl.TransferFunction(nump, denp)

# Cálculo dos parâmetros PID pelo método ITAE
Kd = 1.75 * omega * 0.26 - 1.26
Kp = 2.15 * (omega**2) * 0.26 - 1
Ki = 0.26 * (omega**3)

# Função de transferência do controlador PID
numpid = [Kd, Kp, Ki]
denpid = [1, 0]
PID = ctrl.TransferFunction(numpid, denpid)

# Sistema em malha fechada sem pré-compensador
sys_closed = ctrl.feedback(PID * Gp)

# Resposta ao degrau do sistema em malha fechada
t = np.linspace(0, 6, 61)
t, yitae = ctrl.step_response(sys_closed, t)

# Plotagem da resposta ao degrau sem pré-compensador
plt.figure()
plt.plot(t, yitae, label='Sem pré-compensador')
plt.grid(True)

# Implementação do pré-compensador
# O pré-compensador é um filtro proporcional baseado no ganho integrativo Ki
numfiltro = [Ki]
denfiltro = [1]
Filtro = ctrl.TransferFunction(numfiltro, denfiltro)

# Sistema final com pré-compensador
sys_final = ctrl.series(Filtro, sys_closed)
t, yitae_final = ctrl.step_response(sys_final, t)

# Plotagem da resposta ao degrau com pré-compensador
plt.plot(t, yitae_final, '-.', label='Com pré-compensador')
plt.title('Resposta a um degrau unitário com o PID ITAE')
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.legend()
plt.show()
