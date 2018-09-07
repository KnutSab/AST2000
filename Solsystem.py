import class_file as cf
from AST2000SolarSystem import AST2000SolarSystem
username = amveddeg
seed = AST2000SolarSystem.get_seed(username)
System = AST2000SolarSystem(seed)

r = 8598366.6945

cl = cf.integral(a = 0,b = 1e-6, m = 3.3474472*(10**(-27)),T = 3e3,my = 0,N = int(1e5),Interval = int(1e3),Tau = (1e-13))
dpdt = cl.PositionVelocityUpdate()[6]
N_box = cl.BoxForceCounter()[2]
dNdt = cl.FuelConsup()
m_fuel = cl.InitialRocketMass()[1]
T_launch = cl.TimeToEscapeVel()
launch_pos = np.array(r,cl.OneDimensionalLaunch()[1][0])
t_launch = 0

dpdt, N_box, dNdt, m_fuel = engine_simulation()
T_launch, launch_pos, t_launch = get_launch_parameters()

System.engine_settings(dpdt, N_box, dNdt, m_fuel, T_launch, launch_pos, t_launch)
final_launch_pos = launch_simulation(T_launch, launch_pos, t_launch)
System.mass_needed_launch(final_launch_pos, test=True)
"""
System.take_picture(picture_name)
dlambda1, dlamda2 = System.measure_doppler_shifts()
dist = System.analyse_distances()
ang, satVel, satPos = orientation(picture_name, dlambda1, dlambda 2, dist)
System.manual_orientation(ang, satVel, satPos)
System.send_satellite(filename="satCommands.txt")
p = get_target_planet()
System.land_on_planet(p, filename="landCommands.txt"
"""
