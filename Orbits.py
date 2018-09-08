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
period = attribute.period #
psi = attribute.psi
radius = attribute.radius
rho0 = attribute.rho0
star_mass = attribute.star_mass
star_radius = attribute.star_radius
temperature = attribute.temperature
vx0 = attribute.vx0
vy0 = attribute.vy0
x0 = attribute.x0
y0 = attribute.y0
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

def plot_orbit(G, m_1, m_2, r, x0, y0): #LAG INTEGRASJONSLØKKE FOR FAEN! (LATE JÆVLER)
    r_vec = x0[0]+y0[0]
    m = G*(m_1+m_2)
    acc = m*r_vec/r
    #leapfrog
