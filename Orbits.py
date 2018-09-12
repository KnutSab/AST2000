from AST2000SolarSystem import AST2000SolarSystem
import numpy as np
import matplotlib.pyplot as plt

#planet_orbit(e,a,)



#planet orbit =
attribute = AST2000SolarSystem(40689)
a = attribute.a #Planetenes halvakser
area_lander = attribute.area_lander #overflateareal lander
area_sat = attribute.area_sat #overflateareal satelitt
e = attribute.e #eksentrisitet planeter
mass = attribute.mass #massen til planetene i solmasser
number_of_planets = attribute.number_of_planets #Antall planeter i solsystemet
omega = attribute.omega #polar angle initialposition of planets
period = attribute.period #Planetenes periode for 1 omløp om seg selv
psi = attribute.psi #Den polare vinkelen aphelion
radius = attribute.radius #Radiuen til planetene
rho0 = attribute.rho0 #Atmosfæretrykk på planetene
star_mass = attribute.star_mass #Massen til stjernen
star_radius = attribute.star_radius #Radiusen til stjernen
temperature = attribute.temperature #Overflatetemperatur til stjernen
vx0 = attribute.vx0 #x-komponent til farten til planetene
vy0 = attribute.vy0 #y-komponent til farten til planetene
x0 = attribute.x0 #x-koordinat for utgangsposisjonen
y0 = attribute.y0 #y-koordinat for utgangsposisjonen

"""

def integrator(x0,y0,vx0,vy0,T,dt):
    N = int(T/dt)
    G = 4*np.pi**2
    ax = np.zeros(N+1)
    ay = np.zeros(N+1)
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0] = x0
    y[0] = y0
    vx = np.zeros(N+1)
    vy = np.zeros(N+1)
    vx[0] = vx0
    vy[0] = vy0
    pos = np.array((x0,y0))
    for i in range(N):
        new_pos[i] =
        ax[i] = (-(G*star_mass)/(x[i]**2 + y[i]**2) * (x[i]/(np.sqrt(x[i]**2 + y[i]**2))))
        ay[i] = (-(G*star_mass)/(x[i]**2 + y[i]**2) * (y[i]/(np.sqrt(x[i]**2 + y[i]**2))))
        vx[i+1] = vx[i] + ax[i]*dt
        vy[i+1] = vy[i] + ay[i]*dt
        x[i+1] = x[i] + vx[i+1]*dt
        y[i+1] = y[i] + vy[i+1]*dt

    return(x,y)

#orbit = integrator(x0[0],y0[0],vx0[0],vy0[0],20,0.00001)
x1 = integrator(x0[0],y0[0],vx0[0],vy0[0],20,0.00001)[0],integrator(x0[0],y0[0],vx0[0],vy0[0],20,0.00001)[1])
x2 = integrator(x0[1],y0[1],vx0[1],vy0[1],20,0.00001)[0],integrator(x0[1],y0[1],vx0[1],vy0[1],20,0.00001)[1])
x3 = integrator(x0[2],y0[2],vx0[2],vy0[2],20,0.00001)[0],integrator(x0[2],y0[2],vx0[2],vy0[2],20,0.00001)[1])
x4 = integrator(x0[3],y0[3],vx0[3],vy0[3],20,0.00001)[0],integrator(x0[3],y0[3],vx0[3],vy0[3],20,0.00001)[1])
x5 = integrator(x0[4],y0[4],vx0[4],vy0[4],20,0.00001)[0],integrator(x0[4],y0[4],vx0[4],vy0[4],20,0.00001)[1])
x6 = integrator(x0[5],y0[5],vx0[5],vy0[5],20,0.00001)[0],integrator(x0[5],y0[5],vx0[5],vy0[5],20,0.00001)[1])
x7 = integrator(x0[6],y0[6],vx0[6],vy0[6],20,0.00001)[0],integrator(x0[6],y0[6],vx0[6],vy0[6],20,0.00001)[1])
#plt.axis("equal")
#plt.plot(orbit[0],orbit[1])
"""
pos0 = np.array((x0,y0))
vel0 = np.array((vx0,vy0))

N_planets = number_of_planets
N_time = 150000
pos = np.zeros((N_time, 2, N_planets))
vel = np.zeros((N_time, 2, N_planets))
a = np.zeros((N_time,2,N_planets))
G = 4*np.pi**2
pos[0] = pos0
vel[0] = vel0
dt = 0.0001
for i in range(N_time-1):
    temp = (-(G*star_mass)/(pos[i,0]**2 + pos[i,1]**2) * (pos[i]/(np.sqrt(pos[i,0]**2 + pos[i,1]**2))))
    print(temp.shape)
    print(a[i].shape)
    a[i] = temp

    vel[i+1] += (vel[i] + a[i]*dt)
    pos[i+1] += (pos[i] + vel[i+1]*dt)
print(pos)
plt.axis("equal")
plt.plot(pos[:,0],pos[:,1])
plt.show()
print(mass)
