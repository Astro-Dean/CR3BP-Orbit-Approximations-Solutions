import numpy as np
from quasi_orbit_approx import third_order_solution_lissajous, plot_orbit_approx
import matplotlib.pyplot as plt

Az = 5000
Ay = 5000
m = 1e-5
tf = 20 * np.pi
num_points = 10000
theta1 = 0
theta2 = 0
date = [10, 19, 2025, 0, 0, 0]  # Month, Day, Year, Hour, Minute, Second

x, y, z = third_order_solution_lissajous(Ay, Az, m, tf, num_points, theta1, theta2, date)
plot_orbit_approx(x,y,z, "halo")
