import numpy as np
#Note that the next import is required for Matplotlib versions before 3.2.0. 
#For versions 3.2.0 and higher, you can plot 3D plots without importing
#from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Rot2D(X, Y, Alpha):#rotates point (X,Y) on angle alpha with respect to Origin
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

def Prizma(x0, y0, a, b):#return lists for a prism
    PX = [x0-4/5*a, x0+(1/5)*a, x0+(1/5)*a, x0-4/5*a]
    PY = [y0+(1/2)*b, y0+(1/2)*b, y0-(1/2)*b, y0+(1/2)*b]
    return PX, PY

def Spring(x0, y0, phi, rad):#return lists for a spring
    SX = [x0+rad*t*sp.cos(t)/(6*math.pi) for t in np.linspace(0, 6*math.pi+(1/2)*math.pi+phi,100)]
    SY = [y0+rad*t*sp.sin(t)/(6*math.pi) for t in np.linspace(0, 6*math.pi+(1/2)*math.pi+phi,100)]
    return SX, SY

#defining parameters
#the angle of the plane (and the prism)
alpha = math.pi/10
# size of the prism
a = 5
b = a*sp.tan(alpha)
#size of the beam 
l = 5

#defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')
#here x, y, Vx, Vy, Wx, Wy, xC are functions of 't'
s = 4*sp.cos(3*t)
phi = 4*sp.sin(t-10)

#Motion of the prism with a spring (translational motion)
Xspr = s*sp.cos(alpha)+4/5*a
Yspr = -s*sp.sin(alpha)-1/2*b

VmodSignPrism = sp.diff(s, t)
VxSpr = VmodSignPrism*sp.cos(alpha)
VySpr = -VmodSignPrism*sp.sin(alpha)

WmodSignPrism = sp.diff(VmodSignPrism, t)
WxSpr = WmodSignPrism*sp.cos(alpha)
WySpr = -WmodSignPrism*sp.sin(alpha)

#Motion of the beam with respect to a spring (A - the farthest point on the beam from the spring)
xA = Xspr-l*sp.sin(phi)
yA = Yspr+l*sp.cos(phi)


omega = sp.diff(phi,t)

VxA = VxSpr - omega*l*sp.cos(phi)
VyA = VySpr - omega*l*sp.sin(phi)

VxArel = - omega*l*sp.cos(phi)
VyArel = - omega*l*sp.sin(phi)

#constructing corresponding arrays
T = np.linspace(0, 20, 1000)
XSpr = np.zeros_like(T)
YSpr = np.zeros_like(T)
VXSpr = np.zeros_like(T)
VYSpr = np.zeros_like(T)
Phi = np.zeros_like(T)
XA = np.zeros_like(T)
YA = np.zeros_like(T)
VXA = np.zeros_like(T)
VYA = np.zeros_like(T)
VXArel = np.zeros_like(T)
VYArel = np.zeros_like(T)
#filling arrays with corresponding values
for i in np.arange(len(T)):
    XSpr[i] = sp.Subs(Xspr, t, T[i])
    YSpr[i] = sp.Subs(Yspr, t, T[i])
    VXSpr[i] = sp.Subs(VxSpr, t, T[i])
    VYSpr[i] = sp.Subs(VySpr, t, T[i])
    Phi[i] = sp.Subs(phi, t, T[i])
    XA[i] = sp.Subs(xA, t, T[i])
    YA[i] = sp.Subs(yA, t, T[i])
    VXA[i] = sp.Subs(VxA, t, T[i])
    VYA[i] = sp.Subs(VyA, t, T[i])
    VXArel[i] = sp.Subs(VxArel, t, T[i])
    VYArel[i] = sp.Subs(VyArel, t, T[i])

#here we start to plot
fig = plt.figure(figsize=(17, 8))

ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')
ax1.set(xlim=[XSpr.min()-2*a, XSpr.max()+2*a], ylim=[YSpr.min()-2*a, YSpr.max()+2*a])

#plotting a plane
ax1.plot([XSpr.min()-a, XSpr.max()+a], [-(XSpr.min()-a)*sp.tan(alpha), -(XSpr.max()+a)*sp.tan(alpha)], 'black')

#plotting initial positions

#plotting a prism
PrX, PrY = Prizma(XSpr[0],YSpr[0],a,b)
Prism = ax1.plot(PrX, PrY, 'black')[0]

#plotting a spring
SpX, SpY = Spring(XSpr[0],YSpr[0], Phi[0], a/8)
Spr, = ax1.plot(SpX, SpY, 'black')

#plotting a beam
Beam, = ax1.plot([XSpr[0], XA[0]], [YSpr[0], YA[0]], 'black')

varphi=np.linspace(0, 2*math.pi, 20)
r=l/10
Point = ax1.plot(XA[0]+r*np.cos(varphi), YA[0]+r*np.sin(varphi),color=[1, 0, 1])[0]

ax2 = fig.add_subplot(4, 2, 2)
ax2.plot(T, VXSpr)

ax2.set_xlabel('T')
ax2.set_ylabel('VXPrizm')

ax3 = fig.add_subplot(4, 2, 4)
ax3.plot(T, VYSpr)

ax3.set_xlabel('T')
ax3.set_ylabel('VYPrizm')

ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(T, VXA)

ax4.set_xlabel('T')
ax4.set_ylabel('VXA')

ax5 = fig.add_subplot(4, 2, 8)
ax5.plot(T, VYA)

ax5.set_xlabel('T')
ax5.set_ylabel('VYA')

plt.subplots_adjust(wspace=0.3, hspace = 0.7)

#function for recounting the positions
def anima(i):
    PrX, PrY = Prizma(XSpr[i],YSpr[i],a,b)
    Prism.set_data(PrX, PrY)
    SpX, SpY = Spring(XSpr[i],YSpr[i], Phi[i], a/8)
    Spr.set_data(SpX, SpY)
    Beam.set_data([XSpr[i], XA[i]], [YSpr[i], YA[i]])
    Point.set_data(XA[i]+r*np.cos(varphi), YA[i]+r*np.sin(varphi))
    return Prism, Spr, Beam, Point

# animation function
anim = FuncAnimation(fig, anima,
                     frames=1000, interval=0.01, blit=True)

plt.show()



