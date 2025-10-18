import numpy as np
import matplotlib.pyplot as plt
from math import floor

def UTC2JDT(M, D, Y, hr, min, sec, msec=0.0):
    if M < 1 or M > 12:
        raise ValueError("Month must be between 1 and 12, inclusive.")
    if D < 1 or D > 31:
        raise ValueError("Day must be between 1 and 31, inclusive.")
    
    A = int((M-12)/12)
    B = 1461*(Y + 4800 + A)
    C = 367*(M - 2 - 12*A)
    E = int((Y + 4900 + A)/100)

    JDN = int(B/4) + int(C/12) - int(3*E/4) + D - 32075
    JDT = JDN + (hr - 12)/24 + min/1400 + sec/86400 + msec/86_400_000_000
    return JDT

def norm_angle(x):
    return x % 360.0

def astro_values(M,D,Y,hr,min,sec):
    JD = UTC2JDT(M,D,Y,hr,min,sec)
    T = (JD - 2451545.0)/36525.0

    eps_prime = norm_angle(280.46646 + 36000.76983*T + 0.0003032*T**2)
    earth_peri = norm_angle(102.93735 + 1.71946*T + 0.00046*T**2)
    sun_peri = norm_angle(earth_peri + 180.0)
    eps = norm_angle(218.3164477 + 481267.88123421*T - 0.0015786*(T**2) + (T**3)/538841.0 - (T**4)/65194000.0)
    lun_peri = norm_angle(83.3532465 + 4069.0137287*T)
    Omega = (125.04452 - 1934.136261*T + 0.0020708*T**2 + T**3/450000)
    return eps, eps_prime, lun_peri, sun_peri, Omega

def third_order_solution_lissajous(Ay, Az, m, T, num_points, theta1, theta2, date):
    """
    Using the third-order solution for quasi-periodic Lissajous orbits
    at L2 Lagranina libration point found by Farquhar and Kamel
    
    Found here:
    Farquhar, R. W., & Kamel, A. A. (1973). Quasi-periodic orbits about the translunar libration point.
    Celestial Mechanics, 7(4), 458-473. https://doi.org/10.1007/BF01227511
    """

    e = 0.054900489         # moons eccentricity
    ep = 0.0167217          # Earths eccentricity
    aap = 0.0025093523      # ratio of the semimajor axes for the orbits of Earth and the Moon
    gamma = 0.0900463066    # tangent of the mean inclination of the Moon's orbit
    wxy0 = 1.865485
    wz0 = 1.794291
    a = 2.000034
    b = 7.436984
    c = 2.204904
    d = 3.219481
    BL = 3.190423657
    CL = 2.659334398
    DL = 2.538009811
    EL = 2.572040953
    k = a*wxy0/(b + wxy0**2)
    t = np.linspace(0, T, num_points)
    M, D, Y, hr, min, sec = date
    eps, eps_prime, omega, omega_prime, Omega0 = np.radians(astro_values(M,D,Y,hr,min,sec))
    
    xi = (1 - m)*t + eps - eps_prime
    phi = (1 - (3/4)*m**2 - (225/32)*m**3)*t + eps - omega
    
    C1 = 0.09210089*(e/m)**2 + 0.02905486*Ay**2 + 0.007644849*Az**2
    wxy2 = 0.1387811*(e/m)**2 + 0.04349909*Ay**2 - 0.04060812*Az**2
    wz2 =  0.5981779*(e/m)**2 - 0.03293845*Ay**2 + 0.03923249*Az**2

    wxy = wxy0/(1 + m**2*wxy2)
    wz = wz0/(1 + m**2*wz2)

    T1 = wxy*t + theta1
    T2 = wz*t + theta2

    x1 = 0.341763*Ay*np.sin(T1)
    y1 = Ay*np.cos(T1)
    z1 = Az*np.sin(T2)

    x2_p1 = 0.554904*(e/m)*Ay*np.sin(phi - T1) + 0.493213*(e/m)*Ay*np.sin(phi + T1)
    x2_p2 = -0.09588405*Ay**2*np.cos(2*T1) + 0.128774 * Az**2*np.cos(2*T2)
    x2_p3 = -0.268186*Az**2 - 0.205537*Ay**2

    x2 = x2_p1 + x2_p2 + x2_p3

    y2_p1 = -1.90554*(e/m)*Ay*np.cos(phi - T1) + 1.210699*(e/m)*Ay*np.cos(phi + T1)
    y2_p2 = -0.055296*Ay**2*np.sin(2*T1) - 0.08659705*Az**2*np.sin(2*T2)

    y2 = y2_p1 + y2_p2

    z2_p1 = 1.052082*(e/m)*Az*np.sin(phi+T2) + 1.856918*(e/m)*Az*np.sin(phi - T2)
    z2_p2 = 0.4241194*Ay*Az*np.cos(T2 - T1) + 0.1339910*Ay*Az*np.cos(T2 + T1)

    z2 = z2_p1 + z2_p2

    x3_p1 = (e/m)**2 * Ay*(-0.122841*np.sin(2*phi - T1) + 0.643204*np.sin(2*phi + T1))
    x3_p2 = (e/m)*Az**2*(0.198388*np.cos(phi) - 0.387184*np.cos(phi - 2*T2) + 0.335398*np.cos(phi + 2*T2))
    x3_p3 = (e/m)*Ay**2*(0.173731 * np.cos(phi) + 0.325999*np.cos(phi - 2*T1) - 0.270466*np.cos(phi + 2*T1))
    x3_p4 = (e/m)*Ay*(-1.10033*np.sin(phi - T1 - 2*xi) - 1.189247*np.sin(phi + 2*T1 - 2*xi))
    x3_p5 = Ay*Az**2*(-0.430448*np.sin(2*T2 - T1) - 0.031302*np.sin(2*T2 + T1))
    x3_p6 = Ay**3*(0.027808*np.sin(3*T1)) + C1*Ay*np.sin(T1)
    x3_p7 = Ay*(-0.38856*np.sin(T1 - 2*xi) + 0.455452*np.sin(T1 + 2*xi))

    x3 = x3_p1 + x3_p2 + x3_p3 + x3_p4 + x3_p5 + x3_p6 + x3_p7

    y3_p1 = (e/m)**2*Ay*(0.608685*np.cos(2*phi - T1) + 1.407026*np.cos(2*phi + T1))
    y3_p2 = (e/m)*Az**2*(-0.116822*np.sin(phi) - 0.214742*np.sin(phi - 2*T2) - 0.232503*np.sin(phi + 2*T2))
    y3_p3 = (e/m)*Ay**2*(-0.109499*np.sin(phi) - 0.144553*np.sin(phi - 2*T1) - 0.155751*np.sin(phi + 2*T1))
    y3_p4 = (e/m)*Ay*(2.733367*np.cos(phi - T1 - 2*xi) - 3.848485*np.cos(phi + T1 - 2*xi))
    y3_p5 = Ay*Az**2*(-1.191421*np.cos(2*T2 - T1) - 0.000165*np.cos(2*T2 + T1))
    y3_p6 = Ay**3*(-0.027574*np.cos(3*T1)) + Ay*(-1.743411*np.cos(T1 - 2*xi) + 0.741825*np.cos(T1 + 2*xi))

    y3 = y3_p1 + y3_p2 + y3_p3 + y3_p4 + y3_p5 + y3_p6

    z3_p1 = (e/m)**2*Az*(-0.536625*np.sin(2*phi - T2) + 1.103381*np.sin(2*phi + T2))
    z3_p2_p = 0.367360*np.cos(phi + T2 + T1) + 0.063629*np.cos(phi - T2 + T1) - 0.034729*np.cos(phi + T2 - T1)
    z3_p2 = (e/m)*Ay*Az*(-0.353754*np.cos(phi - T2 - T1) + z3_p2_p)
    z3_p3 = (e/m)*Az*(-2.353465*np.sin(phi - T2 - 2*xi) - 3.831413*np.sin(phi + T2 - 2*xi))
    z3_p4 = Az**3*(0.017664*np.sin(3*T2)) + Az*Ay**2*(-0.86684*np.sin(T2 - 2*T1) - 0.044724*np.sin(T2 + 2*T1))
    z3_p5 = Az*(-1.487917*np.sin(T2 - 2*xi) + 0.475507*np.sin(T2 + 2*xi))

    z3 = z3_p1 + z3_p2 + z3_p3 + z3_p4 + z3_p5

    x = m*x1 + m**2*x2 + m**3*x3
    y = m*y1 + m**2*y2 + m**3*y3
    z = m*z1 + m**2*z2 + m**3*z3

    return x, y, z

if __name__ == "__main__":
    Ay = Az = 3500
    m = 1e-5
    T = 20*np.pi
    num_points = 10000
    theta1 = theta2 = 0
    date = [10, 18, 2025, 0, 0, 0]
    x, y, z = third_order_solution_lissajous(Ay, Az, m, T, num_points, theta1, theta2, date)
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
    ax.scatter(0,0,0, "k*", label="L2 Point")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
