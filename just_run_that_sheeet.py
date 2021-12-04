import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

r = 90  * (np.pi/180)
t = 50000

plt.figure(figsize=(10, 10))
fig = plt.figure()
ax = fig.gca(projection = 'polar')
ax.plot(r, t, color ='b', marker = 'o', markersize = '3')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

line1, = ax.plot([0, 0],[0,t/np.sqrt(2)], color = 'r', linewidth = 10)
line2, = ax.plot([np.pi/2, np.pi/2],[0,t/np.sqrt(2)], color = 'r', linewidth = 10)
line3, = ax.plot([np.pi, np.pi],[0,t/np.sqrt(2)], color = 'r', linewidth = 10)
line4, = ax.plot([3*np.pi/2, 3*np.pi/2],[0,t/np.sqrt(2)], color = 'r', linewidth = 10)
line11, = ax.plot([0, 7*np.pi/4],[t/np.sqrt(2),t], color = 'r', linewidth = 10)
line21, = ax.plot([np.pi/2, np.pi/4],[t/np.sqrt(2),t], color = 'r', linewidth = 10)
line31, = ax.plot([np.pi, 3*np.pi/4],[t/np.sqrt(2),t], color = 'r', linewidth = 10)
line41, = ax.plot([3*np.pi/2, 5*np.pi/4],[t/np.sqrt(2),t], color = 'r', linewidth = 10)


def update(angle):
    line1.set_data([angle, angle], [0,t/np.sqrt(2)])
    line2.set_data([np.pi/2 + angle, np.pi/2 + angle], [0,t/np.sqrt(2)])
    line3.set_data([np.pi + angle, np.pi + angle], [0,t/np.sqrt(2)])
    line4.set_data([3*np.pi/2 + angle, 3*np.pi/2 + angle], [0,t/np.sqrt(2)])
    line11.set_data([0 + angle, 7*np.pi/4 + angle],[t/np.sqrt(2),t])
    line21.set_data([np.pi/2 + angle, np.pi/4 + angle],[t/np.sqrt(2),t])
    line31.set_data([np.pi + angle, 3*np.pi/4 + angle],[t/np.sqrt(2),t])
    line41.set_data([3*np.pi/2 + angle, 5*np.pi/4 + angle],[t/np.sqrt(2),t])
    return line1, line2, line3, line4, line11, line21, line31, line41,

frames = np.linspace(0,2*np.pi,120)

ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=.01)

plt.show()
