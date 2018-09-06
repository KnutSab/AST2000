import class_file as cf
import numpy as np
import matplotlib.pyplot as plt
import numba

np.random.seed(69)
"""
Box = cf.integral(0,1e-6,3.3474472*(10**(-27)),3e3,0,1000,int(1e5),(1e-13))
posisjon = Box.PositionVelocityUpdate()
#print(posisjon)
"""
Box = cf.integral(0,1e-6,3.3474472*(10**(-27)),3e3,0,int(1e5),int(1e3),(1e-9))
#posisjon = Box.PositionVelocityUpdate()
FA = Box.BoxForceCounter()
#print("Total Force: {:g}, Number of Chambers: {:g}".format(FA[0], FA[1]))
#Ost = Box.FuelConsup()
print(FA)
