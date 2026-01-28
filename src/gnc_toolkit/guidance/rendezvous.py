import numpy as np

def solve_lambert(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float = 398600.4418, tm: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves Lambert's problem using a Universal Variable formulation.
    Finds v1 and v2 given r1, r2, and time of flight dt.

    Args:
        r1 (np.ndarray): Initial position vector (km).
        r2 (np.ndarray): Final position vector (km).
        dt (float): Time of flight (s).
        mu (float): Gravitational parameter.
        tm (int): Transfer mode (+1 for short way, -1 for long way). Default +1.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - v1 (np.ndarray): Velocity at r1 (km/s).
            - v2 (np.ndarray): Velocity at r2 (km/s).
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    # Cosine of the change in true anomaly
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    
    # Determine the change in true anomaly (dnu)
    # The arccos function returns a value in [0, pi].
    dnu = np.arccos(cos_dnu)

    # tm = 1  => Short Way (Delta Nu < 180 degrees)
    # tm = -1 => Long Way (Delta Nu > 180 degrees)
    
    if tm == 1:
        # Short way: dnu is already in [0, pi]
        pass
    else:
        # Long way: dnu should be in [pi, 2pi]
        dnu = 2 * np.pi - dnu

    # Calculate constant A
    # A = sin(dnu) * sqrt(r1*r2 / (1 - cos(dnu)))
    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / (1.0 - cos_dnu))
    
    # Check for singularity (180 degree transfer)
    if A == 0.0:
        raise ValueError("Lambert Solver: A=0 (Cannot solve for 180 deg transfer directly).")

    # Iteration variables for Universal Variable formulation
    psi = 0.0
    c2 = 1.0/2.0
    c3 = 1.0/6.0
    
    max_iter = 100
    tol = 1e-6
    
    for _ in range(max_iter):
        y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(c2)
        
        if A > 0.0 and y < 0.0:
            # Readjust psi to avoid negative y (which implies elliptic domain error)
             while y < 0.0:
                 psi += 0.1
                 # Recalculate c2, c3
                 if psi > 1e-6:
                     sq_psi = np.sqrt(psi)
                     c2 = (1 - np.cos(sq_psi)) / psi
                     c3 = (sq_psi - np.sin(sq_psi)) / (sq_psi**3)
                 y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(c2)
            
        if y == 0: y = 1e-10
        
        chi = np.sqrt(y / c2)
        dt_new = (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)
        
        if abs(dt - dt_new) < tol:
            break
            
        # Stumpff functions update for next iteration
        if psi > 1e-6:
            sq_psi = np.sqrt(psi)
            c2 = (1 - np.cos(sq_psi)) / psi
            c3 = (sq_psi - np.sin(sq_psi)) / (sq_psi**3)
        elif psi < -1e-6:
            sq_psi = np.sqrt(-psi)
            c2 = (1 - np.cosh(sq_psi)) / psi
            c3 = (np.sinh(sq_psi) - sq_psi) / (np.sqrt(-psi)**3)
        else:
            c2 = 1.0/2.0
            c3 = 1.0/6.0
            
        # Recalculate y with updated c2, c3 (refinement)
        y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(c2)
        chi = np.sqrt(y / c2)
        
        # Newton-Raphson Derivative d(dt)/d(psi)
        term1 = chi**3 * (c2 - 1.5*c3)
        term2 = 0.125 * A * (3*c3*chi/np.sqrt(c2) + A*np.sqrt(c2/y))
        dtdpsi = (term1 + term2) / np.sqrt(mu)
        
        if dtdpsi == 0.0: dtdpsi = 1.0
        
        # Update psi
        psi += (dt - dt_new) / dtdpsi
        
    # Calculate velocities
    f = 1.0 - y / r1_mag
    g = A * np.sqrt(y/mu)
    g_dot = 1.0 - y / r2_mag
    
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2


def cw_equations(r0: np.ndarray, v0: np.ndarray, n: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagates relative state using Clohessy-Wiltshire (Hill's) equations.
    Assumes circular target orbit.
    
    Args:
        r0 (np.ndarray): Initial relative position [x, y, z] (km).
        v0 (np.ndarray): Initial relative velocity [vx, vy, vz] (km/s).
        n (float): Mean motion of target orbit (rad/s).
        t (float): Time to propagate (s).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - r_t (np.ndarray): Relative position at time t.
            - v_t (np.ndarray): Relative velocity at time t.
    """
    x, y, z = r0
    vx, vy, vz = v0
    
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)
    
    # CW matrix propagation
    # x: Radial, y: Along-track, z: Cross-track
    
    # Position
    xt = (4 - 3*c)*x + (s/n)*vx + (2/n)*(1-c)*vy
    yt = (6*(s - nt))*x + y + (2/n)*(c-1)*vx + (4*s/n - 3*t)*vy
    zt = c*z + (s/n)*vz
    
    # Velocity
    vxt = (3*n*s)*x + c*vx + (2*s)*vy
    vyt = (6*n*(c-1))*x + (-2*s)*vx + (4*c - 3)*vy
    vzt = -n*s*z + c*vz
    
    return np.array([xt, yt, zt]), np.array([vxt, vyt, vzt])

def cw_targeting(r0: np.ndarray, r_target: np.ndarray, t: float, n: float) -> np.ndarray:
    """
    Calculates the initial velocity v0 required to reach r_target from r0 in time t.
    (Two-impulse rendezvous first burn).
    
    Args:
        r0 (np.ndarray): Initial relative position [x, y, z].
        r_target (np.ndarray): Target relative position at time t.
        t (float): Transfer time (s).
        n (float): Mean motion (rad/s).
        
    Returns:
        np.ndarray: Required initial velocity v0.
    """
    # R(t) = Phi_rr * r0 + Phi_rv * v0
    # v0 = Phi_rv^-1 * (R(t) - Phi_rr * r0)
    
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)
    
    # Phi_rr components (from cw_equations)
    # x_row = [(4-3c), 0, 0]
    # y_row = [6(s-nt), 1, 0]
    # z_row = [0, 0, c]
    
    # This is coupled (x, y) and decoupled (z)
    
    # Z component (decoupled)
    # zt = c*z0 + (s/n)*vz0  => vz0 = (zt - c*z0) * (n/s)
    z0 = r0[2]
    zt = r_target[2]
    if abs(s) < 1e-6:
        # Singularity at t = k * Period / 2
        vz0 = 0.0 # Cannot solve or requires impulse elsewhere
    else:
        vz0 = (zt - c*z0) * n / s
        
    # In-plane (x, y)
    # xt - (4-3c)x0 = (s/n)vx0 + (2/n)(1-c)vy0
    # yt - (6(s-nt)x0 + y0) = (2/n)(c-1)vx0 + (4s/n - 3t)vy0
    
    # Let A be matrix for v:
    # A = [ s/n      2(1-c)/n ]
    #     [ 2(c-1)/n 4s/n - 3t]
    
    dx = r_target[0] - (4 - 3*c)*r0[0]
    dy = r_target[1] - (6*(s - nt)*r0[0] + r0[1])
    
    # Determinant of A
    # det = (s/n)(4s/n - 3t) - (2(1-c)/n)(2(c-1)/n)
    #     = (1/n^2) [ 4s^2 - 3nts - 4(1-c)(c-1) ]
    #     = (1/n^2) [ 4s^2 - 3nts + 4(1-c)^2 ]
    #     = ...
    
    # Using numpy linear solver for 2x2
    A = np.array([
        [s/n, (2/n)*(1-c)],
        [(2/n)*(c-1), (4*s/n - 3*t)]
    ])
    
    b = np.array([dx, dy])
    
    try:
        v_xy = np.linalg.solve(A, b)
        vx0, vy0 = v_xy
    except np.linalg.LinAlgError:
        vx0, vy0 = 0.0, 0.0 # Singularity
        
    return np.array([vx0, vy0, vz0])
