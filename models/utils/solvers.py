"""

Implementation of several ODE Solvers (i.e. Runge-Kutta, Euler)

"""


import torch.nn as nn
import torch as th

        
def integrate(func, y0, t, method):
    
    solution = th.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
    solution[0] = y0

    j = 1
    y0_temp = y0 #(eta[t], u[t], v[t]), (32x32x3)
    for t0, t1 in zip(t[:-1], t[1:]):
        dt = t1 - t0

        if method == "rk4":
            dy_velo = rk4_step_func(func, t0, dt, t1, y0_temp, eq='velo')
        elif method == "euler":
            f0_velo = func(dt, y0_temp, eq='velo')  # (eta[t], du[t+1]/dt, dv[t+1]/dt)
            dy_velo = dt * f0_velo[..., 1:]

        y1_velo = th.cat((y0_temp[...,0].unsqueeze(-1),
                          y0_temp[..., 1:] + dy_velo), dim=-1)  # (eta[t], u[t+1], v[t+1])
        
        if method == "rk4":
            dy_eta = rk4_step_func(func, t0, dt, t1, y1_velo, eq='eta')
        elif method == "euler":
            f0_eta = func(t0, y1_velo, eq='eta') #(deta[t+1]/dt)
            dy_eta = dt * f0_eta

        y1 = th.cat(((y0_temp[...,0] + dy_eta).unsqueeze(-1), y1_velo[...,1:]), dim=-1) # (eta[t+1], u[t+1], v[t+1])

        solution[j] = y1

        y0_temp = y1
        j += 1

    return solution
    

# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6

def rk4_step_func(func, t0, dt, t1, y0, eq, f0=None):
    k1 = f0
    
    if k1 is None:
        k1 = func(t0, y0, eq) # (eta[t+1]/dt) when eq='eta' and (eta[t], u[t+1]/dt, v[t+1]/dt) when eq='velo'
    
    half_dt = dt * 0.5
    
    if eq == 'eta':
        # Propagate only over eta
        k2 = func(t0 + half_dt, y0 + half_dt * th.cat((k1.unsqueeze(-1),
                                                       th.zeros_like(y0[...,1:])), dim=-1), eq)
        k3 = func(t0 + half_dt, y0 + half_dt * th.cat((k2.unsqueeze(-1),
                                                       th.zeros_like(y0[...,1:])), dim=-1), eq)
        k4 = func(t1, y0 + dt * th.cat((k3.unsqueeze(-1),
                                                       th.zeros_like(y0[...,1:])), dim=-1), eq)
        
        return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth
        
    elif eq == 'velo':
        # Propagate over velocity functions u and v
        k2 = func(t0 + half_dt, y0 + half_dt * th.cat((th.zeros_like(y0[...,0]).unsqueeze(-1),
                                                       k1[...,1:]), dim=-1), eq)
        k3 = func(t0 + half_dt, y0 + half_dt * th.cat((th.zeros_like(y0[...,0]).unsqueeze(-1),
                                                       k2[...,1:]), dim=-1), eq)
        k4 = func(t0 + half_dt, y0 + half_dt * th.cat((th.zeros_like(y0[...,0]).unsqueeze(-1),
                                                       k3[...,1:]), dim=-1), eq)

        return (k1[...,1:] + 2 * (k2[...,1:] + k3[...,1:]) + k4[...,1:]) * dt * _one_sixth
        
    


def rk4_alt_step_func(func, t0, dt, t1, y0, f0=None):
    """Smaller error with slightly more compute."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3))
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125

