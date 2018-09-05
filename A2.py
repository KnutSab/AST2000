import class_file as cf
import numpy as np
import matplotlib.pyplot as plt

print("A.2.1:")
vp_value = cf.integral(-2.5*10**4,2.5*10**4,3.3474472*10**(-27),10**4,0)
plotting_graph = cf.plotting(vp_value.vel_prob_dist()[0],vp_value.vel_prob_dist()[1],r"speed of particles $[m/s]$",r"velocity probability density","velocity distrobution")
plotting_graph.plot()

print("---------------------------------------------------------------")
print("A.2.2:")
int_value = cf.integral(5*10**3,30*10**3,3.3474472*10**(-27),10**4,0)
value = int_value.int_solver_gauss()
print("The probability is %g" %(value))
print("The number of particles per volume with speed v_x is %g" %(value*10**5))

print("---------------------------------------------------------------")
print("A.2.3:")
vp_value = cf.integral(0*10**4,3*10**4,3.3474472*10**(-27),10**4,0)
plotting_graph = cf.plotting(vp_value.absolute_v()[0],vp_value.absolute_v()[1],r"speed of particles $[m/s]$",r"velocity probability density","velocity distrobution")
plotting_graph.plot()

print("---------------------------------------------------------------")
print("A.2.2:")
int_value = cf.integral(5*10**3,30*10**3,3.3474472*10**(-27),10**4,0)
value = int_value.int_solver_MB()
print("The probability is %g" %(value))
#print("The number of particles per volume with speed v_x is %g" %(value*10**5))
