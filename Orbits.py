from AST2000SolarSystem import AST2000SolarSystem
import numpy as np
import matplotlib.pyplot as plt
def planet_orbit(e,a,theta,sigma):
    f = theta - sigma
    r = a*(1-e**2)/(1+e*np.cos(f))
    return r
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
print(period)

#print(vx0[0], vy0)
"""
#print(x0)
plt.plot(0,0,'bo')
plt.plot(x0[0],y0[0], 'ro')
plt.plot(x0[1],y0[1], 'go')
plt.plot(x0[2],y0[2], 'co')
plt.plot(x0[3],y0[3], 'mo')
plt.plot(x0[4],y0[4], 'ko')
plt.plot(x0[5],y0[5], 'ro')
plt.plot(x0[6],y0[6], 'yo')
plt.show()
"""
def integrator(x0,v0,T,dt):
    G = 6.67*10e-11
    def R(t):
        F = -((G*star_mass)/(np.sqrt(x0**2 + y0**2)**3))*(x0+y0)
        return F
    N = int(T/dt)
    t = np.zeros(N)
    x = x0
    v = v0
    a_i = R(t)
    for i in range(N):
        a_iplus1 = R(t)
    for i in range(N):
        t += dt
        x += v*dt + 0.5* a_i*dt**2
        a_iplus1 = R(t)
        v += 0.5*(a_i + a_iplus1)*dt
        a_i = a_iplus1
    return x
pos = integrator(x0,y0,10,0.001)
print(pos)
plt.plot(pos)
plt.show()
