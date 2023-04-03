
"""Packages"""

import numpy as np
import matplotlib.pyplot as plt

Earth_radius = 6370e3

"""
#######################################################################
Old Functions (taken from util.py)
#######################################################################
"""

def cyclogeostrophy(ug, vg, coriolis, lon, lat, epsilon):
    """Compute velocities from cyclogeostrophic approximation using the iterative method used in Penven et al. (2014)
    •ug: u component of geostrophic velocity
    •vg: v component of geostrophic velocity
    •coriolis: Coriolis parameter
    •lon: x coordinates (lon)
    •lat: y coordinates (lat)
    •epsilon: residual"""
    u_cg = np.copy(ug)
    v_cg = np.copy(vg)
    mask = np.zeros_like(ug)
    errsq = 1000*np.ones_like(ug)
    arreps = epsilon * np.ones_like(ug)
    n_iter = 0
    while np.any(mask == 0) and n_iter < 100:
        n_iter += 1
        u_n = np.copy(u_cg)
        v_n = np.copy(v_cg)
        errsq_n = np.copy(errsq)
        advec_u, advec_v = advection(u_n,v_n,lon,lat)
        u_np1 = ug - (1/coriolis)*advec_v
        v_np1 = vg + (1/coriolis)*advec_u
        errsq = np.square(u_np1-u_n) + np.square(v_np1-v_n)
        #print('Iteration process', 'n_iter:', n_iter, 'ug:', u_v[20,25], 'u_cg:', u_cg[20,25], 'errsq_e:', errsq_e[20,25], 'errsq:',errsq[20,25])
        mask_np1 = np.where(errsq < arreps, 1, 0)
        mask_n = np.where(errsq > errsq_n, 1, 0)
        u_cg = mask * u_n + (1-mask) * ( mask_n * u_n + (1-mask_n) * u_np1 )
        v_cg = mask * v_n + (1-mask) * ( mask_n * v_n + (1-mask_n) * v_np1 )
        mask = np.maximum(mask, np.maximum(mask_n, mask_np1))
        print((n_iter, np.where(mask==1)[0].shape, np.max(errsq)))

        #mask = np.where( (errsq < arreps) | (errsq>errsq_e), 0, 1) # elementwise OR condition
        #errsq = np.where( mask == 0, 0, errsq )
        #maxerr = np.max(errsq)
        #print('mask:',mask[20,25],'errsq_cor:',errsq[20,25],'max_error:',maxerr)
        #print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
    return u_cg, v_cg, mask



"""
#######################################################################
New implemented Functions
#######################################################################
"""

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
    a = lon[2:size,2:size]
    b = lat[2:size,2:size]
    c = lon_u[:size,1:size]
    d = lat_u[:size,1:size]
    e = lon_v[1:size,:size]
    f = lat_v[1:size,:size]
    
    fig = plt.subplot()
    for i in range(size):
        fig.plot(c[:,i-1],d[:,i-1], color = 'black', linewidth=2, zorder=1)
        fig.plot(e[i-1,:],f[i-1,:], color = 'black', linewidth=2, zorder=2)
        
    for i in range(size-2):
        fig.plot(c[i+1,:],d[i+1,:], '--', color = 'black', linewidth=1, zorder=1)
        fig.plot(e[:,i+1],f[:,i+1], '--', color = 'black', linewidth=1, zorder=2)
        
    fig.scatter(a,b, s=60,label='SSH', zorder=4)
    fig.scatter(c,d, s=60, label='x direction velocity', zorder=3)
    fig.scatter(e,f, s=60, label='y direction velocity', zorder=3)
    fig.set_xlabel('longitude [degrees]')
    fig.set_ylabel('latitude [degrees]')
    fig.legend(loc=1, framealpha =1)
    return fig

def compute_derivatives(field, lon, lat, axis):
    """
    Computes the x and y derivatives of a 2D field using finite differences.
    
    Arguments:
    field - a 2D array representing the field to be differentiated;
    lon: longitude (x direction);
    lat: latitude (y direction);
    axis - 0: derivate in regards of x, 1:  derivate in regards of y;
    """
    if axis == 0:
        f = field[:, 1:] - field[:, :-1]
        dx = (lon[:, 1:] - lon[:, :-1]) * np.pi / 180     # in radian
        lat = (lat[:, 1:] + lat[:, :-1])/2
        dx = dx * Earth_radius * np.cos(lat*np.pi/180)
        dfdx = f/dx
        #dfdx = np.c_[dfdx, np.zeros(len(dfdx))]
        
    if axis == 1:
        f = field[:-1, :] - field[1:, :]
        dx = (lat[:-1, :] - lat[1:, :]) * np.pi / 180     # in radian
        dx = dx * Earth_radius
        dfdx = f/dx
        #dfdx = np.r_[dfdx, np.zeros([1,np.size(dfdx,1)])]
    return dfdx

def compute_gradient(field, lon, lat):
    """
    Function that computes the gradient of a field
    
    Arguments:
    field - a 2D array representing the field to be differentiated;
    lon: longitude (x direction);
    lat: latitude (y direction);
    """
    fx, fy = compute_derivatives(field, lon, lat, 0), compute_derivatives(field, lon, lat, 1)
    return fx, fy

def compute_advection_u(u, v, lon, lat):
    """
    Function that computes the advection term for a velocity field in the direction x
    The function also interpolate the values to a v point
    
    Arguments:
    u - velocity in x direction;
    v - velocity in y direction;
    lon: longitude (x direction);
    lat: latitude (y direction);
    """
    u_adv = u
    v_adv = v
    
    dudx = compute_derivatives(u_adv, lon, lat, 0)
    dudx = interpolate_y(dudx)
    
    dudy = compute_derivatives(u_adv, lon, lat, 1)
    dudy = interpolate_x(dudy)
    
    u_adv = interpolate_x(u_adv)
    u_adv = interpolate_y(u_adv)
    
    v_adv = v_adv[1:-1,:]
    
    adv_u = u_adv * dudx + v_adv * dudy
    return adv_u

def compute_advection_v(u, v, lon, lat):
    """
    Function that computes the advection term for a velocity field in the direction x
    The function also interpolate the values to a u point
    
    Arguments:
    u - velocity in x direction;
    v - velocity in y direction;
    lon: longitude (x direction);
    lat: latitude (y direction);
    """
    u_adv = u
    v_adv = v
    
    dvdx = compute_derivatives(v_adv, lon, lat, 0)
    dvdx = interpolate_y(dvdx)
    
    dvdy = compute_derivatives(v_adv, lon, lat, 1)
    dvdy = interpolate_x(dvdy)
    
    v_adv = interpolate_x(v_adv)
    v_adv = interpolate_y(v_adv)
    
    u_adv = u_adv[1:-1,:]
    
    adv_v = v_adv * dvdx + v_adv * dvdy
    return adv_v

def interpolate_y(field):
    """
    Function to interpolate the values of a field in the y direction
    
    Arguments:
    field - field to be interpolated
    """
    f = field
    
    f_y = np.zeros([np.size(f,0)-1,np.size(f,1)])
    
    for i in range(0, np.size(f,0)-1):
        f_y[i,:] = (f[i,:] + f[i+1,:])/2

    return f_y

def interpolate_x(field):
    """
    Function to interpolate the values of a field in the x direction
    
    Arguments:
    field - field to be interpolated
    """
    f = field
    
    f_x = np.zeros([np.size(f,0),np.size(f,1)-1])
    
    for i in range(0, np.size(f,1)-1):
        f_x[:,i] = (f[:,i] + f[:,i+1])/2

    return f_x
