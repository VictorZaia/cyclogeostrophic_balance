
"""Packages"""

import numpy as np
import Tools.geometry as geo

"""Input data"""
gravity = 9.81

# =============================================================================
# Geostrophy
# =============================================================================

def geostrophy(ssh, dx_ssh, dy_ssh, coriolis_factor_u, coriolis_factor_v):
    
    # Computing the gradient of the ssh
    grad_ssh_x, grad_ssh_y = geo.compute_gradient(ssh, dx_ssh, dy_ssh)
    
    # Interpolation of the data (moving the grad into the u and v position)
    grad_ssh_y = geo.interpolate(grad_ssh_y, axis=0)
    grad_ssh_y = geo.interpolate(grad_ssh_y, axis=1)
    
    grad_ssh_x = geo.interpolate(grad_ssh_x, axis=1)
    grad_ssh_x = geo.interpolate(grad_ssh_x, axis=0)
        
    # Interpolating the coriolis
    cu = geo.interpolate(coriolis_factor_u, axis=0)
    cu  = geo.interpolate(cu, axis=1)
    
    cv = geo.interpolate(coriolis_factor_v, axis=1)
    cv = geo.interpolate(cv, axis=0)
    
    # Computing the geostrophic velocities
    u_geos = - gravity * grad_ssh_y / cu
    v_geos = gravity * grad_ssh_x / cv
    
    return u_geos, v_geos
