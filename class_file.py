import numpy as np
import matplotlib.pyplot as plt
import random as random
class integral:
    def __init__(self,a,b,m,T,my,N,Interval,Tau):
        self.a = a
        self.b = b
        self.m = m
        self.my = my
        self.T = T
        self.k = 1.38064852*(10**(-23))
        self.N = N
        self.Interval = Interval
        self.Tau = Tau
    def int_solver_gauss(self):
        "Calculates the integral of f, using the midtpoint method"
        a = self.a
        b = self.b
        m = self.m
        my = self.my
        T = self.T
        k = self.k

        sigma = np.sqrt((k*T)/m)
        N = 10000 #number of steps
        x_array = np.linspace(a,b,N) #x-values
        dx = (b-a)/N
        int_array = np.zeros(len(x_array))
        for i,j in zip(x_array,range(len(int_array))):
            "Calculates the area of each rectangle and adds it to int_array"
            f_start = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(1/2)*((i-my)/sigma)**2)
            f_end = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(1/2)*(((i+dx)-my)/sigma)**2)
            int_array[j] = ((f_start+f_end)/2)*dx
        return (np.sum(int_array)) #sums up all rectangles to approch integral, probability for a particle to have v_x

    def vel_prob_dist(self):
        "Calculates the velocity probability distrobution"
        a = self.a
        b = self.b
        m = self.m
        my = self.my
        T = self.T
        k = self.k

        sigma = np.sqrt((k*T)/m)
        N = 10**5 #number of steps
        x_array = np.linspace(a,b,N) #x-values
        dx = (b-a)/N #small step along x-axis
        fx_array = np.zeros(len(x_array)) #empty array
        for i,j in zip(x_array,range(len(fx_array))):
            "Calculates the f(x) values, probability density"
            f_value = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(1/2)*((i-my)/sigma)**2)
            fx_array[j] = f_value
        return x_array,fx_array

    def absolute_v(self):
        "Calculates absolute velocity with the non-gaussian Maxwell Boltzmann"
        a = self.a
        b = self.b
        m = self.m
        my = self.my
        T = self.T
        k = self.k

        sigma = np.sqrt((k*T)/m) #sigma value for Maxwell Boltzmann
        N = 10**5 #number of steps
        x_array = np.linspace(a,b,N) #x-values
        dx = (b-a)/N #small step along x-axis
        fx_array = np.zeros(len(x_array)) #empty array
        for i,j in zip(x_array,range(len(fx_array))):
            "Calculates the f(x) values, probability density"
            f_value = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(1/2)*((i-my)/sigma)**2)*4*np.pi*(i**2)*dx
            fx_array[j] = f_value
        return x_array,fx_array
    def int_solver_MB(self):
        "Calculates the integral of f, using the midtpoint method, with Maxwell-Boltzmann"
        a = self.a
        b = self.b
        m = self.m
        my = self.my
        T = self.T
        k = self.k

        sigma = np.sqrt((k*T)/m)
        N = 10000 #number of steps
        x_array = np.linspace(a,b,N) #x-values
        dx = (b-a)/N
        int_array = np.zeros(len(x_array))
        for i,j in zip(x_array,range(len(int_array))):
            "Calculates the area of each rectangle and adds it to int_array"
            f_start = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(1/2)*((i-my)/sigma)**2)*4*np.pi*(i**2)*dx
            f_end = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(1/2)*(((i+dx)-my)/sigma)**2)*4*np.pi*(i**2)*dx
            int_array[j] = ((f_start+f_end)/2)*dx
        return (np.sum(x_array*int_array)) #sums up all rectangles to approch integral, probability for a particle to have v_x
    def ParticlePosition(self):
        a = self.a
        b = self.b
        N = self.N
        x_position = [random.uniform(a, b) for i in range(N)]
        y_position = [random.uniform(a, b) for i in range(N)]
        z_position = [random.uniform(a, b) for i in range(N)]
        return x_position,y_position,z_position
    def ParticleVelocity(self):
        k = self.k
        T = self.T
        m = self.m
        N = self.N
        my = (0)
        sigma = np.sqrt((k*T)/m)
        vel_x = np.zeros(N)
        vel_y = np.zeros(N)
        vel_z = np.zeros(N)
        for i in range(N):
            vel_x[i] = random.gauss(self.my,sigma)
            vel_y[i] = random.gauss(self.my,sigma)
            vel_z[i] = random.gauss(self.my,sigma)
        return vel_x,vel_y,vel_z
    def PositionVelocityUpdate(self):
        N = self.N
        Interval = self.Interval
        Tau = self.Tau
        a = self.a
        b = self.b
        m = self.m
        Box = integral(0,1e-6,3.3474472*10**(-27),3e5,0,1000,int(1e5),1e-13)
        NewPositionX = Box.ParticlePosition()[0]
        NewPositionY = Box.ParticlePosition()[1]
        NewPositionZ = Box.ParticlePosition()[2]
        NewVelocityX = Box.ParticleVelocity()[0]
        NewVelocityY = Box.ParticleVelocity()[1]
        NewVelocityZ = Box.ParticleVelocity()[2]
        F = 0
        Esc = 0

        for n in range(Interval):
            NewPositionX = NewPositionX+NewVelocityX*Tau
            NewPositionY = NewPositionY+NewVelocityY*Tau
            NewPositionZ = NewPositionZ+NewVelocityZ*Tau
            #NewVelocityX[NewPositionX > b]

            for i in range(N):
                if NewPositionX[i] >= 0.25*b and NewPositionX[i] <= (0.75*b) and NewPositionY[i] >= 0.25*b and NewPositionY[i] <= (0.75*b):
                    F = (m*(abs(NewVelocityZ[i])))/Tau
                    Esc += 1

                if NewPositionX[i] >= b or NewPositionX[i] <= a:
                    NewVelocityX[i] = NewVelocityX[i]*(-1)
                elif NewPositionY[i] >= b or NewPositionY[i] <= a:
                    NewVelocityY[i] = NewVelocityY[i]*(-1)
                elif NewPositionZ[i] >= b or NewPositionZ[i] <= a:
                    NewVelocityZ[i] = NewVelocityZ[i]*(-1)
            #print(NewPositionX[0])
        return NewPositionX,NewPositionY,NewPositionZ,NewVelocityX,NewVelocityY,NewVelocityZ
class GetData(integral):
    def __init__(self,):

    def get_data()


class plotting:
    "plotting class"
    def __init__(self,x,fx,xlabel,ylabel,graph_text):
        self.x = x
        self.fx = fx
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.graph_text = graph_text

    def plot(self):
        x = self.x
        fx = self.fx
        plt.plot(x,fx, label = self.graph_text)
        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()
#lol
