"""Auxiliary functions used in several notebooks of the repo."""

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# Geophysical parameter: Earth radius
Earth_radius = 6370e3     # in meters

def set_boundaries_to_zero(field):
    f = np.copy(field)
    f[0,:]  = f[-1,:] = f[:,0]  = f[:,-1] = 0
    return f

def set_boundaries_to(field, field_b):
    f = np.copy(field)
    f[0,:], f[-1,:], f[:,0], f[:,-1] = field_b[0,:], field_b[-1,:], field_b[:,0], field_b[:,-1]
    return f


def von_neuman_euler(field, axis=None):
    """Apply Von Neuman boundary conditions to the field."""
    f = np.copy(field)
    if axis == 0 or axis == None:
        f[0,:]  = f[1,:]
        f[-1,:] = f[-2,:]
    if axis == 1 or axis == None:
        f[:,0]  = f[:,1]
        f[:,-1] = f[:,-2]
    return f

def derivative(field, lon, lat, axis):
    """Compute partial derivative along longitude using second-order centered scheme.
    field: field to derive
    lon, lat: longitude and latitude arrays, same size as field
    axis: along which derivation is carried out.
    axis = 0 corresponds to latitude, axis = 1 corresponds to longitude.
    The output is in field_unit/meters.
    """
    f = np.roll(field, -1, axis=axis) - field
    if axis == 0:
        dx = ( np.roll(lat, -1, axis=0) - lat ) * np.pi / 180     # in radian
        dx = dx * Earth_radius
    if axis == 1:
        dx = ( np.roll(lon, -1, axis=1) - lon ) * np.pi / 180     # in radian
        dx = dx * Earth_radius * np.cos(lat*np.pi/180)
    f = f / dx
    f = von_neuman_euler(f, axis=axis)
    return f

def gradient(field, lon, lat):
    """Compute gradient of input scalar field."""
    fx, fy = derivative(field, lon, lat, axis=1), derivative(field, lon, lat, axis=0)
    return fx, fy

def divergence(u, v, lon, lat):
    """Compute divergence of a 2D-vector with components u, v."""
    f = derivative(u, lon, lat, axis=1) + derivative(v, lon, lat, axis=0)
    return f

def rotational(u, v, lon, lat):
    """Compute vertical component of rotational of a 2D-vector with components u, v."""
    f = derivative(v, lon, lat, axis=1) - derivative(u, lon, lat, axis=0)
    return f

def laplacian(field, lon, lat):
    """Compute Laplacian of field."""
    dx = 0.5 * ( np.roll(lat, -1, axis=0) - np.roll(lat, 1, axis=0) ) * np.pi / 180     # in radian
    dx = dx * Earth_radius
    dy = 0.5 * ( np.roll(lon, -1, axis=1) - np.roll(lon, 1, axis=1) ) * np.pi / 180     # in radian
    dy = dy * Earth_radius * np.cos(lat*np.pi/180)
    fx = np.roll(field, -1, axis=0) -2 * field + np.roll(field, 1, axis=0)
    fy = np.roll(field, -1, axis=1) -2 * field + np.roll(field, 1, axis=1)
    f = fx/(dx*dx) + fy/(dy*dy)
    f = von_neuman_euler(f)
    return f

def advection(u, v, lon, lat):
    #u2, v2, uv = u*u, v*v, u*v
    #adv_u = derivative(u2, lon, lat, axis=1) + derivative(uv, lon, lat, axis=0)
    #adv_v = derivative(uv, lon, lat, axis=1) + derivative(v2, lon, lat, axis=0)
    adv_u = u * derivative(u, lon, lat, axis=1) + v * derivative(u, lon, lat, axis=0)
    adv_v = u * derivative(v, lon, lat, axis=1) + v * derivative(v, lon, lat, axis=0)
    return adv_u, adv_v



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

def derivative_jax(field, axis):
    dfdx = 0
    if axis == 0 or axis == None:
        dfdx = grad(field)
    elif axis ==1:
        dfdx = grad(field, argnums=(1))
    else:
        print('Error: define the right variable to compute the derivate')
    return dfdx

def gradient_jax(field):
    fx, fy = derivative_jax(field, lon, lat, axis=1), derivative(field, lon, lat, axis=0)
    return fx, fy
