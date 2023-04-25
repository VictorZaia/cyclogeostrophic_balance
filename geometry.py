
"""Packages"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

"""Input data"""

Earth_radius = 6370e3
p0 = np.pi/180

# =============================================================================
# Functions without JAX
# =============================================================================

def neuman_forward(field, axis=0):
    """
    Apply Von Neuman boundary conditions to the field.
    """
    f = np.copy(field)
    if axis == 0:
        f[-1,:] = field[-2, :]
    if axis == 1:
        f[:,-1] = field[:,-2]
    return f

def compute_spatial_steps(lon, lat, bounds=(1e2,1e4), fill_value=1e12):
    """
    Compute dx and dy spatial steps of a grid defined by lon, lat.
    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of cos and arccos functions
    to avoid truncation issues.
    Inputs: lon, lat: 2D arrays [ny,nx]
            bounds: range of acceptable values. out of this range, set to fill_value
    Outputs: dx[ny,nx-1], dy[ny-1,nx]: 2D arrays
    """
    dx, dy = np.zeros_like(lon), np.zeros_like(lon)
    # dx
    dlat, dlon = p0*(lat[:,1:]-lat[:,:-1]), p0*(lon[:,1:]-lon[:,:-1])
    dx[:,:-1] = Earth_radius*np.sqrt( dlat**2 + np.cos(p0*lat[:,:-1]) * np.cos(p0*lat[:,1:]) * dlon**2 )
    dx = neuman_forward(dx, axis=1)
    dx = np.where(dx>bounds[0], dx, fill_value)       # avoid zero or huge steps due to
    dx = np.where(dx<bounds[1], dx, fill_value)       # spurious zero values in lon lat arrays
    # dy
    dlat, dlon = p0*(lat[1:,:]-lat[:-1,:]), p0*(lon[1:,:]-lon[:-1,:])
    dy[:-1,:] = Earth_radius*np.sqrt( dlat**2 + np.cos(p0*lat[:-1,:]) * np.cos(p0*lat[1:,:]) * dlon**2 )
    dy = neuman_forward(dy, axis=0)
    dy = np.where(dy>bounds[0], dy, fill_value)
    dy = np.where(dy<bounds[1], dy, fill_value)
    return dx, dy


def interpolate(field, axis=0):
    """
    Function to interpolate the values of a field in the y direction
    Arguments:
    field - field to be interpolated
    """
    f = np.copy(field)
    if axis == 0:
        f[:-1,:] = 0.5 * ( field[:-1,:] + field[1:,:] )
    if axis == 1:
        f[:,:-1] = 0.5 * ( field[:,:-1] + field[:,1:] )
    f = neuman_forward(f, axis=axis)
    return f


def compute_derivative(field, dxy, axis=0):
    """
    Computes the x or y derivatives of a 2D field using finite differences.   
    Arguments:
    field: a 2D array representing the field to be differentiated. Size (ny, nx);
    dxy: 2D array of spatial steps.
    axis: 'x' or 'y'
    """
    f = np.copy(field)
    if axis == 0:
        f[:-1,:] = field[1:, :] - field[:-1, :]
    if axis == 1:
        f[:,:-1] = field[:, 1:] - field[:, :-1]
    f = neuman_forward(f, axis=axis)
    return f/dxy

def compute_gradient(field, dx, dy):
    """
    Function that computes the gradient of a field  
    Arguments:
    field: a 2D array representing the field to be differentiated;
    dx: spatial steps in x direction;
    dy: spatial steps in y direction;
    """
    fx, fy = compute_derivative(field, dx, axis=1), compute_derivative(field, dy, axis=0)
    return fx, fy


def compute_advection_u(u, v, dx, dy):
    """
    Function that computes the advection term for a velocity field in the direction x
    The function also interpolate the values to a v point 
    Arguments:
    u - velocity in x direction;
    v - velocity in y direction;
    dx: spatial steps in x direction;
    dy: spatial steps in y direction;
    """
    u_adv = np.copy(u)
    v_adv = np.copy(v)
    
    dudx = compute_derivative(u, dx, axis=1)    # h points
    dudx = interpolate(dudx, axis=0)            # v points
    
    dudy = compute_derivative(u, dy, axis=0)    # vorticity points
    dudy = interpolate(dudy, axis=1)            # v points
    
    u_adv = interpolate(u_adv, axis=1)          # h points
    u_adv = interpolate(u_adv, axis=0)          # v points
    
    adv_u = u_adv * dudx + v_adv * dudy         # v points
    return adv_u

def compute_advection_v(u, v, dx, dy):
    """
    Function that computes the advection term for a velocity field in the direction x
    The function also interpolate the values to a u point
    Arguments:
    u - velocity in x direction;
    v - velocity in y direction;
    dx: spatial steps in x direction;
    dy: spatial steps in y direction;
    """
    u_adv = np.copy(u)
    v_adv = np.copy(v)
    
    dvdx = compute_derivative(v, dx, axis=1)   # vorticity points
    dvdx = interpolate(dvdx, axis=0)           # u points
    
    dvdy = compute_derivative(v, dy, axis=0)   # h points
    dvdy = interpolate(dvdy, axis=1)           # u points
    
    v_adv = interpolate(v_adv, axis=1)         # vorticity points
    v_adv = interpolate(v_adv, axis=0)         # u points
    
    adv_v = u_adv * dvdx + v_adv * dvdy        # u points
    return adv_v

# =============================================================================
# Functions with JAX
# =============================================================================

def neuman_forward_jax(field, axis=0):
    f = jnp.copy(field)
    if axis == 0:
        f = f.at[-1,:].set(field[-2, :])
    if axis == 1:
        f = f.at[:,-1].set(field[:,-2])
    return f

def interpolate_jax(field, axis=0):
    """
    Function to interpolate the values of a field in the y direction
    Arguments:
    field - field to be interpolated
    """
    f = jnp.copy(field)
    if axis == 0:
        f = f.at[:-1,:].set(0.5 * (field[:-1,:] + field[1:,:]))
    if axis == 1:
        f = f.at[:,:-1].set(0.5 * (field[:,:-1] + field[:,1:]))
    f = neuman_forward_jax(f, axis=axis)
    return f

def compute_derivative_jax(field, dxy, axis=0):
    """
    Computes the x or y derivatives of a 2D field using finite differences.   
    Arguments:
    field: a 2D array representing the field to be differentiated. Size (ny, nx);
    dxy: 2D array of spatial steps.
    axis: 'x' or 'y'
    """
    f = jnp.copy(field)
    if axis == 0:
        f = f.at[:-1,:].set(field[1:, :] - field[:-1, :])
    if axis == 1:
        f= f.at[:,:-1].set(field[:, 1:] - field[:, :-1])
    f = neuman_forward_jax(f, axis=axis)
    return f/dxy

def compute_gradient_jax(field, dx, dy):
    """
    Function that computes the gradient of a field  
    Arguments:
    field: a 2D array representing the field to be differentiated;
    dx: spatial steps in x direction;
    dy: spatial steps in y direction;
    """
    fx, fy = compute_derivative_jax(field, dx, axis=1), compute_derivative_jax(field, dy, axis=0)
    return fx, fy

def compute_advection_u_jax(u, v, dx, dy):
    """
    Function that computes the advection term for a velocity field in the direction x
    The function also interpolate the values to a v point 
    Arguments:
    u - velocity in x direction;
    v - velocity in y direction;
    dx: spatial steps in x direction;
    dy: spatial steps in y direction;
    """
    u_adv = jnp.copy(u)
    v_adv = jnp.copy(v)
    
    dudx = compute_derivative_jax(u, dx, axis=1)    # h points
    dudx = interpolate_jax(dudx, axis=0)            # v points
    
    dudy = compute_derivative_jax(u, dy, axis=0)    # vorticity points
    dudy = interpolate_jax(dudy, axis=1)            # v points
    
    u_adv = interpolate_jax(u_adv, axis=1)          # h points
    u_adv = interpolate_jax(u_adv, axis=0)          # v points
    
    adv_u = u_adv * dudx + v_adv * dudy         # v points
    return adv_u

def compute_advection_v_jax(u, v, dx, dy):
    """
    Function that computes the advection term for a velocity field in the direction x
    The function also interpolate the values to a u point
    Arguments:
    u - velocity in x direction;
    v - velocity in y direction;
    dx: spatial steps in x direction;
    dy: spatial steps in y direction;
    """
    u_adv = jnp.copy(u)
    v_adv = jnp.copy(v)
    
    dvdx = compute_derivative_jax(v, dx, axis=1)   # vorticity points
    dvdx = interpolate_jax(dvdx, axis=0)           # u points
    
    dvdy = compute_derivative_jax(v, dy, axis=0)   # h points
    dvdy = interpolate_jax(dvdy, axis=1)           # u points
    
    v_adv = interpolate_jax(v_adv, axis=1)         # vorticity points
    v_adv = interpolate_jax(v_adv, axis=0)         # u points
    
    adv_v = u_adv * dvdx + v_adv * dvdy        # u points
    return adv_v

# =============================================================================
# Independent functions
# =============================================================================

def PlotGrid(lon,lat,lon_u,lat_u,lon_v,lat_v):
    """
    Function that splits the data and plot the C grid
    
    Arguments:
    lon - longitude of the ssh
    lat - latitude of the ssh
    lon_u - longitude of the velocity in x direction
    lat_u - latitude of the velocity in x direction
    lon_v - longitude of the velocity in y direction
    lat_v - latitude of the velocity in y direction
    """
    size = 5 # reduced size of the mesh to be ploted
    
    "Spliting the mesh"
    a = lon[:size,:size]
    b = lat[:size,:size]
    c = lon_u[:size,:size]
    d = lat_u[:size,:size]
    e = lon_v[:size,:size]
    f = lat_v[:size,:size]
    
    fig = plt.subplot()
    for i in range(size):
        fig.plot(c[:,i],d[:,i], color = 'black', linewidth=2, zorder=1)
        fig.plot(e[i,:],f[i,:], color = 'black', linewidth=2, zorder=2)
        
    for i in range(size):
        fig.plot(c[i,:],d[i,:], '--', color = 'black', linewidth=1, zorder=1)
        fig.plot(e[:,i],f[:,i], '--', color = 'black', linewidth=1, zorder=2)
        
    fig.scatter(a,b, s=60,label='SSH', zorder=4)
    fig.scatter(c,d, s=60, label='x direction velocity', zorder=3)
    fig.scatter(e,f, s=60, label='y direction velocity', zorder=3)
    fig.set_xlabel('longitude [degrees]')
    fig.set_ylabel('latitude [degrees]')
    fig.legend(loc=1, framealpha =1)
    return fig

def compute_spatial_steps(lon, lat, bounds=(1e2,1e4), fill_value=1e12):
    """
    Compute dx and dy spatial steps of a grid defined by lon, lat.
    It makes use of the distance-on-a-sphere formula with Taylor expansion approximations of cos and arccos functions
    to avoid truncation issues.
    Inputs: lon, lat: 2D arrays [ny,nx]
            bounds: range of acceptable values. out of this range, set to fill_value
    Outputs: dx[ny,nx-1], dy[ny-1,nx]: 2D arrays
    """
    dx, dy = np.zeros_like(lon), np.zeros_like(lon)
    # dx
    dlat, dlon = p0*(lat[:,1:]-lat[:,:-1]), p0*(lon[:,1:]-lon[:,:-1])
    dx[:,:-1] = Earth_radius*np.sqrt( dlat**2 + np.cos(p0*lat[:,:-1]) * np.cos(p0*lat[:,1:]) * dlon**2 )
    dx = neuman_forward(dx, axis=1)
    dx = np.where(dx>bounds[0], dx, fill_value)       # avoid zero or huge steps due to
    dx = np.where(dx<bounds[1], dx, fill_value)       # spurious zero values in lon lat arrays
    # dy
    dlat, dlon = p0*(lat[1:,:]-lat[:-1,:]), p0*(lon[1:,:]-lon[:-1,:])
    dy[:-1,:] = Earth_radius*np.sqrt( dlat**2 + np.cos(p0*lat[:-1,:]) * np.cos(p0*lat[1:,:]) * dlon**2 )
    dy = neuman_forward(dy, axis=0)
    dy = np.where(dy>bounds[0], dy, fill_value)
    dy = np.where(dy<bounds[1], dy, fill_value)
    return dx, dy
