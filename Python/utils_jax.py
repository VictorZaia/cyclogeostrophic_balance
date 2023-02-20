# Packages
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from functools import partial

# Input data:
Earth_radius = 6370e3     # in meters

@partial(jit, static_argnums=(1))
def von_neuman_euler(field, axis):
    f = jnp.copy(field)
    if axis == 0:
        f = f.at[0,:].set(f.at[1,:].get())
        f = f.at[-1,:].set(f.at[-2,:].get())
    if axis == 1:
        f = f.at[:,0].set(f.at[:,1].get())
        f = f.at[:,-1].set(f.at[:,-2].get())
    return f

@partial(jit, static_argnums=(3))
def derivative(field, lon, lat, axis):
    f = jnp.roll(field, -1, axis=axis) - field
    if axis == 0:
        dx = ( jnp.roll(lat, -1, axis=0) - lat ) * jnp.pi / 180     # in radian
        dx = dx * Earth_radius
    if axis == 1:
        dx = ( jnp.roll(lon, -1, axis=1) - lon ) * jnp.pi / 180     # in radian
        dx = dx * Earth_radius * jnp.cos(lat*np.pi/180)
    f = f / dx
    f = von_neuman_euler(f, axis=axis)
    return f


def gradient(field, lon, lat):
    fx, fy = derivative(field, lon, lat, axis=1), derivative(field, lon, lat, axis=0)
    return fx, fy

gradient_jit = jit(gradient)