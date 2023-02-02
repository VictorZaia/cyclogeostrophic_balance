
"""
Inversion of the Cyclogeostrophic balance with JAX

@author: victor
"""

"Packages"
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import utils as ut

"Path of the data to be analysed"
path = 'D:/UGA/2 semestre/Research project/data/Old_data/'
file_name = 'eNATL60GULFSTREAM-BLB002_y2009m08d01_hr.nc'
data = xr.open_dataset(path+file_name)
print(data)


"Reading the data"
lon = data.nav_lon.values
lat = data.nav_lat.values 
SSH = data.sossheig.values[0,:,:]
#velocities calculated by a DNS calculation. They will be used to validate the computation of the geostrophic and cyclogeostrophic balance.
u = data.sozocrtx.values[0,:,:]
v = data.somecrty.values[0,:,:]


"Input data"
g = 9.81 #[m/s2]
coriolis = 2 * 7.2722e-05 * np.sin(lat * np.pi / 180)


"Ploting functions"
def _plot(ax,var,title):
    ax.set_title(title)
    im = ax.pcolormesh(lon,lat,var,shading='auto',cmap = "jet")
    plt.colorbar(im,ax=ax)

def plot(u,v,ssh):
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
    ax1.set_ylabel('latitude')
    for ax in (ax1,ax2,ax3):
        ax.set_xlabel('longitude')
    _plot(ax1,u,'Field 1')
    _plot(ax2,v,'Field 2')
    _plot(ax3,ssh,'Field 3')
    return fig


fig = plot(u,v,SSH)

"Computing the geostrophic balance"
gradx_ssh, grady_ssh = ut.gradient(SSH, lon, lat)
u_geos = - g * grady_ssh / coriolis
v_geos =   g * gradx_ssh / coriolis

"Plotting the geostrophic balance"
fig = plot(u,u_geos,u - u_geos)
fig = plot(v,v_geos,v - v_geos)
