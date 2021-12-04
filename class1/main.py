import numpy as np
from matplotlib.animation import FuncAnimation
import sympy as sp
import matplotlib.pyplot as plt

R = 2
omega = 2

t = sp.Symbol('t')

x = R * (omega * t - sp.sin(omega * t))
y = R * (1 - sp.cos(omega * t))
vx = sp.diff(x, t)
vy = sp.diff(y, t)


T = np.linspace(0, 10, 1000)

X = np.zeros_like(T)
Y = np.zeros_like(T)
VY = np.zeros_like(T)
VX = np.zeros_like(T)


for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(vx, t, T[i])
    VY[i] = sp.Subs(vy, t, T[i])

fig = plt.figure()
pl1 = fig.add_subplot(1, 1, 1)
pl1.axis('equal')
pl1.plot(X, Y)
point, = pl1.plot(X[0], X[0], 'r', marker='o')
v_line, = plt.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]])

def animation(i):
    point.set_data(X[i], Y[i])
    v_line.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    return point, v_line,

anime = FuncAnimation(fig, animation, frames=1000, interval=2)

#plt.savefig("mygraph_motion.png")
plt.show()