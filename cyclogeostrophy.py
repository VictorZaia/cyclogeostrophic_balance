
"""Packages"""

import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from tqdm import tqdm

import geometry as geo

# =============================================================================
# Iterative method
# =============================================================================

def cyclogeostrophy(ug, vg, coriolis_u, coriolis_v, lon_u, lat_u, lon_v, lat_v, epsilon):
    """
    Compute velocities from cyclogeostrophic approximation using the iterative method used in Penven et al. (2014)
    ug: u component of geostrophic velocity
    vg: v component of geostrophic velocity
    coriolis: Coriolis parameter
    lon: x coordinates (lon)
    lat: y coordinates (lat)
    epsilon: residual
    """
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
        
        dx_u, dy_u = geo.compute_spatial_steps(lon_u, lat_u)
        dx_v, dy_v = geo.compute_spatial_steps(lon_v, lat_v)
        
        advec_v = geo.compute_advection_v(ug, vg, dx_v, dy_v)
        advec_u = geo.compute_advection_u(ug, vg, dx_u, dy_u)
        
        u_jnp1 = ug - (1/coriolis_u)*advec_v
        v_jnp1 = vg + (1/coriolis_v)*advec_u
        
        errsq = np.square(u_jnp1-u_n) + np.square(v_jnp1-v_n)
        #print('Iteration process', 'n_iter:', n_iter, 'ug:', u_v[20,25], 'u_cg:', u_cg[20,25], 'errsq_e:', errsq_e[20,25], 'errsq:',errsq[20,25])
        mask_jnp1 = np.where(errsq < arreps, 1, 0)
        mask_n = np.where(errsq > errsq_n, 1, 0)
        u_cg = mask * u_n + (1-mask) * ( mask_n * u_n + (1-mask_n) * u_jnp1 )
        v_cg = mask * v_n + (1-mask) * ( mask_n * v_n + (1-mask_n) * v_jnp1 )
        mask = np.maximum(mask, np.maximum(mask_n, mask_jnp1))
        print((n_iter, np.where(mask==1)[0].shape, np.max(errsq)))

        #mask = np.where( (errsq < arreps) | (errsq>errsq_e), 0, 1) # elementwise OR condition
        #errsq = np.where( mask == 0, 0, errsq )
        #maxerr = np.max(errsq)
        #print('mask:',mask[20,25],'errsq_cor:',errsq[20,25],'max_error:',maxerr)
        #print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
    return u_cg, v_cg

# =============================================================================
# Minimization method using a loss function
# =============================================================================

"""Function without JAX"""
def gradient_descent(f, x_init, y_init, learning_rate = 0.01, num_iterations = 1000):

    x, y = x_init, y_init
    
    # calculate the gradient of f at (x, y)
    grad_x = grad(f)
    grad_y = grad(f,argnums=(1))
    
    for i in range(num_iterations):
        # update x and y using gradient descent
        x = x - learning_rate * grad_x(x,y)
        y = y - learning_rate * grad_y(x,y)

    # print the final value of the loss at (x, y)
    print("iteration {}: f(x, y) = {}".format(i, f(x, y)))
        
    return x, y

"""Function using JAX"""
@partial(jit,static_argnums=(0))
def iteration(f,x,y):
    learning_rate = 0.005
    
    x_n = jnp.copy(x)
    y_n = jnp.copy(y)
    
    grad_x = grad(f)
    grad_y = grad(f,argnums=(1))
    
    x_n = x - learning_rate * grad_x(x,y)
    y_n = y - learning_rate * grad_y(x,y)
    
    return x_n,y_n

def gradient_descent_jax(f, x_init, y_init, num_iterations = 2000):
    
    x, y = x_init, y_init
    it=0
        
    while it < num_iterations:
        it+=1
        
        # update x and y using gradient descent
        x,y = iteration(f,x,y)

    # print the final value of the loss at (x, y)
    print("iteration {}: loss(x, y) = {}".format(it, f(x, y)))   
    return x, y
