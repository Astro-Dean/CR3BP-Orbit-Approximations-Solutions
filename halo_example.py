import numpy as np
from quasi_orbit_approx import fourth_order_solution_halo
import matplotlib.pyplot as plt

Az = 38000
m = 1e-11
tf = 20 * np.pi
num_points = 10000
theta = 0

date = [10, 19, 2025, 0, 0, 0]  # Month, Day, Year, Hour, Minute, Second

x, y, z = fourth_order_solution_halo(Az, m, tf, num_points, theta, date)

plt.figure(figsize=(8,8))
plt.plot(x, y)
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()

plt.figure(figsize=(8,8))
plt.plot(x, z)
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Z")
plt.grid()

plt.figure(figsize=(8,8))
plt.plot(y, z)
plt.axis('equal')
plt.xlabel("Y")
plt.ylabel("Z")
plt.grid()

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot(x,y,z, "b")
ax.scatter(0,0, 0, color = "black", marker="*", label="L2 Point")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
