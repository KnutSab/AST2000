from AST2000SolarSystem import AST2000SolarSystem
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

#planet_orbit(e,a,)
start_time = time.time()



#planet orbit =
attribute = AST2000SolarSystem(40689)
a_p = attribute.a #Planetenes halvakser
area_lander = attribute.area_lander #overflateareal lander
area_sat = attribute.area_sat #overflateareal satelitt
e = attribute.e #eksentrisitet planeter
mass = attribute.mass #massen til planetene i solmasser
vx0 = attribute.vx0 #x-komponent til farten til planetene
vy0 = attribute.vy0 #y-komponent til farten til planetene
x0 = attribute.x0 #x-koordinat for utgangsposisjonen
y0 = attribute.y0 #y-koordinat for utgangsposisjonen

star_mass = attribute.star_mass #Massen til stjernen
number_of_planets = attribute.number_of_planets
omega = attribute.omega #polar angle initialposition of planets
period = attribute.period #Planetenes periode for 1 omløp om seg selv
psi = attribute.psi #Den polare vinkelen aphelion
radius = attribute.radius #Radiuen til planetene
rho0 = attribute.rho0 #Atmosfæretrykk på planetene
star_radius = attribute.star_radius #Radiusen til stjernen
temperature = attribute.temperature #Overflatetemperatur til stjernen
#print(star_mass)


#N_time = 200000


#T = antall år på bane * Antall år ønslket
#N = Tidssteg * T
#N_time = T*N







def LeapFrog(x0,y0):
    pos0 = np.array((x0,y0))
    vel0 = np.array((vx0,vy0))
    N_planets = number_of_planets
    T = 20
    N = 200000
    N_time = int(T*N)
    dt = 1/200000
    pos = np.zeros((N_time+1, 2, N_planets))
    vel = np.zeros((N_time+1, 2, N_planets))
    pos[0] = pos0
    vel[0] = vel0
    a = np.zeros((N_time+1,2,N_planets))
    G = 4*np.pi**2
    a[0] = -(((G*star_mass)/(np.sqrt(pos[0,0]**2 + pos[0,1]**2)**3)) * (pos[0]))

    @jit(nopython=True)
    def loop(pos,vel,a,G,N,T,N_time,dt):
        p = 1
        for i in range(N_time):
            pos[i+1] = pos[i] + vel[i]*dt + 0.5*a[i]*(dt**2)
            a[i+1] = -(((G*star_mass)/(np.sqrt(pos[i+1,0]**2 + pos[i+1,1]**2)**3)) * (pos[i+1]))
            vel[i+1] = vel[i] + 0.5*(a[i]+a[i+1])*dt

            if p == 1:
                if i/N_time > 0.1:
                    print("10%")
                    p = 2

            if p == 2:
                if i/N_time > 0.2:
                    print("20%")
                    p = 3
            if p == 3:
                if i/N_time > 0.3:
                    print("30%")
                    p = 4
            if p == 4:
                if i/N_time > 0.4:
                    print("40%")
                    p = 5
            if p == 5:
                if i/N_time > 0.5:
                    print("50%")
                    p = 6
            if p == 6:
                if i/N_time > 0.6:
                    print("60%")
                    p = 7
            if p == 7:
                if i/N_time > 0.7:
                    print("70%")
                    p = 8
            if p == 8:
                if i/N_time > 0.8:
                    print("80%")
                    p = 9
            if p == 9:
                if i/N_time > 0.9:
                    print("90%")
                    p = 10
            if p == 10:
                if i/N_time > 1:
                    print("FINISHED")
                    p = 11
        return(pos,N,T)
    dick = loop(pos,vel,a,G,N,T,N_time,dt)
    pos2 = np.zeros((2, N_planets,N_time+1))
    #print(dick[0])
    @jit(nopython=True)
    def rearrange(pos2,dick0,N_time,N_planets):

        for i in range(N_time):
            for j in range(N_planets):
                pos2[0,j,i] = pos[i,0,j]
                pos2[1,j,i] = pos[i,1,j]
        return(pos2)
    pas2 = rearrange(pos2,dick[0],N_time,N_planets)
    return(pos,pas2,T,N)

#print(pos2)
Leap = LeapFrog(x0,y0)
print("--- %s seconds ---" % (time.time() - start_time))
pos = Leap[0]
pos2 = Leap[1]
T = Leap[2]
N = Leap[3]

attribute.check_planet_positions(Leap[1],(T),(N),writeFile=True)
[planet_positions,times] = np.load('planet_positions.npy')
#attribute.orbit_xml(planet_positions[:,:,::100],times[::100])
#os.system('say "Program fullfoert"')
"""
plt.axis("equal")
plt.plot(pos[:,0],pos[:,1])
plt.show()
"""
