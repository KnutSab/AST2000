from AST2000SolarSystem import AST2000SolarSystem
from orbits import positions
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

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
d = positions(20, 2*10**5,x0,y0)
p = d.LeapFrog()
r = p[0]
print(r.shape)
class satellite:

    def __init__(self, init_time, init_pos, init_vel, N, N_length):
        self.init_time = init_time #initial time to launch
        self.init_pos = np.asarray((init_pos)) #position you launch from
        self.init_vel = np.asarray((init_vel)) #velocity you launch at
        self.N = N #Number of time steps
        self.N_length = N_length #time step length

    def trajectory(self):
        N = self.N
        r_sat = np.zeros((N,2)) #satelite position
        a = np.zeros((N,2)) #acceleration
        vel = np.zeros((N,2)) #velocity
        t = np.zeros(N) #time
        r_sat[0,:] = self.init_pos #setting start position
        vel[0,:] = self.init_vel #setting initial speed
        t[0] = self.init_time #setting initial time
        G = 9.81
        M = mass
        dt = 0.1
        N_length = self.N_length

        for j in range(N):
            force_sum = 0
            for i in range(6):
                force = G*M[i]*(r_sat[j]-r[j,:,i])/abs(r_sat[j]-r[j,:,i])**3
                force_sum += force

            a[0] = -G*star_mass*(r_sat[j])/abs(r_sat[j])**3 - force_sum #acceleration by Newtons second law of motion

            r_sat[i+1] = r_sat[i] + vel[i]*dt + 0.5*a[i]*(dt**2) #using leapfrog
            a[i+1] = -(((G*star_mass)/(np.sqrt(r_sat[i+1,0]**2 + r_sat[i+1,1]**2)**3)) * (r_sat[i+1]))
            vel[i+1] = vel[i] + 0.5*(a[i]+a[i+1])*dt
            t[i+1] += dt
            return t[-1], r_sat[-1], vel[-1]

k = satellite(0, (x0[0]+(radius[0]/150*10**6),y0[0]+(radius[0]/150*10**6)), (10,10), 1000, 0.1)
o = k.trajectory()

print(o)
