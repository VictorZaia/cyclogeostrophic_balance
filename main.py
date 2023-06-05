"""Importing packages"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import tkinter as tk
import os

import Tools.geometry as geo  
import Tools.geostrophy as geost
import Tools.cyclogeostrophy as cyclo
import PreProcessor as pre



"""Reading the input data"""

print("---------------Reading the files---------------")

"Path to the data"
dir_data = os.path.dirname(__file__) + '/'

# =============================================================================
# Manual input of the path to the data
# =============================================================================
# name_mask = 'mask_eNATL60MEDWEST_3.6.nc'
# name_ssh = 'eNATL60MEDWEST-BLB002_y2009m07d01.1h_sossheig.nc'
# name_u = 'eNATL60MEDWEST-BLB002_y2009m07d01.1h_sozocrtx.nc'
# name_v = 'eNATL60MEDWEST-BLB002_y2009m07d01.1h_somecrty.nc'

"opening the data"

ds_ssh = pre.openFile("Select the sea surface height file") # Automatically opens a file dialog
# ds_ssh = xr.open_dataset(dir_data+name_ssh) # Manual input of the path to the data
lon_ssh = ds_ssh.nav_lon.values
lat_ssh = ds_ssh.nav_lat.values
ssh = ds_ssh.sossheig[0].values

ds_u = pre.openFile("Select the u velocity file") # Automatically opens a file dialog
# ds_u = xr.open_dataset(dir_data+name_u) # Manual input of the path to the data
lon_u = ds_u.nav_lon.values
lat_u = ds_u.nav_lat.values
uvel = ds_u.sozocrtx[0].values

ds_v = pre.openFile("Select the v velocity file") # Automatically opens a file dialog
# ds_v = xr.open_dataset(dir_data+name_v) # Manual input of the path to the data
lon_v = ds_v.nav_lon.values
lat_v = ds_v.nav_lat.values
vvel = ds_v.somecrty[0].values

ds_mask = pre.openFile("Select the mask file") # Automatically opens a file dialog
# ds_mask = xr.open_dataset(dir_data+name_mask) # Manual input of the path to the data
mask_ssh = ds_mask.tmask[0,0].values
mask_u = ds_mask.umask[0,0].values
mask_v = ds_mask.vmask[0,0].values

"masking the data"

mask_u = 1 - mask_u
mask_v = 1 - mask_v
mask_ssh = 1- mask_ssh

uvel = ma.masked_array(uvel, mask_u)
vvel = ma.masked_array(vvel, mask_v)
ssh = ma.masked_array(ssh, mask_ssh)
lon_u = ma.masked_array(lon_u, mask_u)
lat_u = ma.masked_array(lat_u, mask_u)
lon_v = ma.masked_array(lon_v, mask_v)
lat_v = ma.masked_array(lat_v, mask_v)
lon_ssh = ma.masked_array(lon_ssh, mask_ssh)
lat_ssh = ma.masked_array(lat_ssh, mask_ssh)

"Creating arrays of spatial steps"

# =============================================================================
# These dx and dy steps must be used to compute derivatives.
# =============================================================================
dx_ssh, dy_ssh = geo.compute_spatial_steps(lon_ssh, lat_ssh)
dx_u, dy_u = geo.compute_spatial_steps(lon_u, lat_u)
dx_v, dy_v = geo.compute_spatial_steps(lon_v, lat_v)

print("---------------Files reading complete---------------")



"""Input parameters"""

gravity = 9.81
coriolis_factor_ssh = 2 * 7.2722e-05 * np.sin(lat_ssh * np.pi / 180)
coriolis_factor_u = 2 * 7.2722e-05 * np.sin(lat_u * np.pi / 180)
coriolis_factor_v = 2 * 7.2722e-05 * np.sin(lat_v * np.pi / 180)



"""Geostrophic balance"""

u_geos, v_geos = geost.geostrophy(ssh, dx_ssh, dy_ssh, coriolis_factor_u, coriolis_factor_v)
u_geos = ma.masked_array(u_geos, mask_u)
v_geos = ma.masked_array(v_geos, mask_v)



"""Cyclogeostrophic balance"""

print("---------------Cyclogeostrophic inversion started---------------")

# =============================================================================
# Input values for the loss function
# =============================================================================
u = np.copy(u_geos.filled(0))
v = np.copy(v_geos.filled(0))

"Defining the cost function"

def loss(u,v):
    """
    Compute the discrepancy between the geostrophic velocities and the cyclogeostrophic velocities
    u: u component of geostrophic velocity
    v: v component of geostrophic velocity
    """
    J_u = np.sum((u + geo.compute_advection_v_jax(u, v, dx_v, dy_v)/coriolis_factor_u.filled(1) - u_geos.filled(0))**2)
    J_v = np.sum((v - geo.compute_advection_u_jax(u, v, dx_u, dy_u)/coriolis_factor_v.filled(1) - v_geos.filled(0))**2)

    return J_u + J_v

print("Initial loss: " + str(loss(u,v)))

"Running the gradient descent method"

print("---------------Starting minimization algorithm---------------")

u_min, v_min = cyclo.gradient_descent_jax(loss,u,v)
u_min = ma.masked_array(u_min, mask_u)
v_min = ma.masked_array(v_min, mask_v)

print("Final loss: " + str(loss(u_min, v_min)))

print("---------------Minimization algorithm done---------------")

"Iterative method"

print("---------------Starting iterative method---------------")

u_it, v_it = cyclo.cyclogeostrophy(u, v, coriolis_factor_u,coriolis_factor_v, lon_u, lat_u, lon_v, lat_v, 0.0001)

print("---------------Iterative method Done---------------")



"""Writing the output data"""

# =============================================================================
# The u and v components will be written as a netcdf file.
# =============================================================================

print("---------------Writing the output data---------------")

array_lat_u = np.asarray(lat_u)
array_lat_v = np.asarray(lat_v)

array_lon_u = np.asarray(lon_u)
array_lon_v = np.asarray(lon_v)


ds_u_out = xr.DataArray(data = u_min, dims = ['x', 'y'], coords = dict(lat = (['x', 'y'], array_lat_u), lon = (['x', 'y'], array_lon_u), mask_u = (['x', 'y'], mask_u)), name = 'Velocity_u')
ds_v_out = xr.DataArray(data = v_min, dims = ['x', 'y'], coords = dict(lat = (['x', 'y'], array_lat_v), lon = (['x', 'y'], array_lon_v), mask_v = (['x', 'y'], mask_v)), name = 'Velocity_v')

ds_u_out.to_netcdf(dir_data + 'u_cyclogeostrophic.nc')
ds_v_out.to_netcdf(dir_data + 'v_cyclogeostrophic.nc')

print("---------------Output data written---------------")

print("---------------Cyclogeostrophic inversion Finished---------------")
