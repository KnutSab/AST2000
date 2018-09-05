import class_file as cf
import random as random
import numpy as np
import matplotlib.pyplot as plt
"""
random.seed(69)

Box = cf.integral(0,1e-6,3.3474472*(10**(-27)),3e3,0,1000,int(1e5),(1e-13))
posisjon = Box.PositionVelocityUpdate()
#print(posisjon)
"""
Box = cf.integral(0,1e-6,3.3474472*(10**(-27)),3e3,0,1000,int(1e5),(1e-13))
posisjon = Box.PositionVelocityUpdate()
