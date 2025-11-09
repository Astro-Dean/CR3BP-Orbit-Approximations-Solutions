import numpy as np
from quasi_orbit_approx import fourth_order_solution_halo, plot_orbit_approx
import matplotlib.pyplot as plt

Az = 34000
m = 1e-11
tf = 40 * np.pi
num_points = 10000
theta = 0

date = [10, 19, 2025, 0, 0, 0]  # Month, Day, Year, Hour, Minute, Second

x, y, z = fourth_order_solution_halo(Az, m, tf, num_points, theta, date)
plot_orbit_approx(x,y,z, "halo")

