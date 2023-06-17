
"""Packages"""

import numpy as np
import numpy.ma as ma

import Tools.geometry as geo

"""Class that will be responsible for storing the data from the netCDF files and the results"""

class Model:

    """Attributes"""

    __coriolis_factor_ssh = None
    __coriolis_factor_u = None
    __coriolis_factor_v = None

    __u_geos = None
    __v_geos = None

    __u_cyclo = None
    __v_cyclo = None

    """Constructors"""

    def __init__(self):
        """
        Constructor that initializes the attributes of the class that will be filled later.
        They were stated as 0, because they will be filled later.
        """
        self.__u = 0
        self.__lat_u = 0
        self.__lon_u = 0
        self.__mask_u = 0

        self.__v = 0
        self.__lat_v = 0
        self.__lon_v = 0
        self.__mask_v = 0
        
        self.__ssh = 0
        self.__lat_ssh = 0
        self.__lon_ssh = 0
        self.__mask_ssh = 0

    """Getters and setters to access the attributes of the class"""

    "u velocity"

    def get_u(self):
        return self.__u
    
    def get_lat_u(self):
        return self.__lat_u
    
    def get_lon_u(self):
        return self.__lon_u
    
    def get_mask_u(self):
        return self.__mask_u
    
    def set_u(self, u_in):
        self.__u = u_in

    def set_lat_u(self, lat_u):
        self.__lat_u = lat_u

    def set_lon_u(self, lon_u):
        self.__lon_u = lon_u
    
    def set_mask_u(self, mask_u):
        self.__mask_u = mask_u

    "v velocity"

    def get_v(self):
        return self.__v
    
    def get_lat_v(self):
        return self.__lat_v
    
    def get_lon_v(self):
        return self.__lon_v
    
    def get_mask_v(self):
        return self.__mask_v
    
    def set_v(self, v_in):
        self.__v = v_in

    def set_lat_v(self, lat_v):
        self.__lat_v = lat_v

    def set_lon_v(self, lon_v):
        self.__lon_v = lon_v
    
    def set_mask_v(self, mask_v):
        self.__mask_v = mask_v
    
    "SSH"

    def get_ssh(self):
        return self.__ssh
    
    def get_lat_ssh(self):
        return self.__lat_ssh
    
    def get_lon_ssh(self):
        return self.__lon_ssh
    
    def get_mask_ssh(self):
        return self.__mask_ssh
    
    def set_ssh(self, ssh):
        self.__ssh = ssh

    def set_lat_ssh(self, lat_ssh):
        self.__lat_ssh = lat_ssh

    def set_lon_ssh(self, lon_ssh):
        self.__lon_ssh = lon_ssh
    
    def set_mask_ssh(self, mask_ssh):
        self.__mask_ssh = mask_ssh

    "Coriolis factor"

    def get_coriolis_factor_ssh(self):
        return self.__coriolis_factor_ssh
    
    def get_coriolis_factor_u(self):
        return self.__coriolis_factor_u
    
    def get_coriolis_factor_v(self):
        return self.__coriolis_factor_v
    
    def set_coriolis_factor_ssh(self, coriolis_factor_ssh):
        self.__coriolis_factor_ssh = coriolis_factor_ssh

    def set_coriolis_factor_u(self, coriolis_factor_u):
        self.__coriolis_factor_u = coriolis_factor_u

    def set_coriolis_factor_v(self, coriolis_factor_v):
        self.__coriolis_factor_v = coriolis_factor_v

    "Output"

    def get_u_geos(self):
        return self.__u_geos
    
    def get_v_geos(self):
        return self.__v_geos

    def get_u_cyclo(self):
        return self.__u_cyclo
    
    def get_v_cyclo(self):
        return self.__v_cyclo

    def set_u_geos(self, u_geos):
        self.__u_geos = u_geos

    def set_v_geos(self, v_geos):
        self.__v_geos = v_geos
    
    def set_u_cyclo(self, u_cyclo):
        self.__u_cyclo = u_cyclo
    
    def set_v_cyclo(self, v_cyclo):
        self.__v_cyclo = v_cyclo
    
    """Methods"""

    def compute_Coriolis_factor(self):
        """
        Function that computes the coriolis factor from the latitude of the measurement points.
        """
        self.__coriolis_factor_ssh = 2 * 7.2722e-05 * np.sin(self.__lat_ssh * np.pi / 180)
        self.__coriolis_factor_u = 2 * 7.2722e-05 * np.sin(self.__lat_u * np.pi / 180)
        self.__coriolis_factor_v = 2 * 7.2722e-05 * np.sin(self.__lat_v * np.pi / 180)

    def mask_data(self):
        """
        Function that masks the data that are not in the domain of the measurements.
        """
        self.__mask_u = 1 - self.__mask_u
        self.__mask_v = 1 - self.__mask_v
        self.__mask_ssh = 1 - self.__mask_ssh

        self.__u = ma.masked_array(self.__u, self.__mask_u)
        self.__v = ma.masked_array(self.__v, self.__mask_v)
        self.__ssh = ma.masked_array(self.__ssh, self.__mask_ssh)
        self.__lon_u = ma.masked_array(self.__lon_u, self.__mask_u)
        self.__lat_u = ma.masked_array(self.__lat_u, self.__mask_u)
        self.__lon_v = ma.masked_array(self.__lon_v, self.__mask_v)
        self.__lat_v = ma.masked_array(self.__lat_v, self.__mask_v)
        self.__lon_ssh = ma.masked_array(self.__lon_ssh, self.__mask_ssh)
        self.__lat_ssh = ma.masked_array(self.__lat_ssh, self.__mask_ssh)
    
    def loss(self, u, v):
        """
        Compute the discrepancy between the geostrophic velocities and the cyclogeostrophic velocities
        u: u component velocity
        v: v component velocity
        """

        dx_u, dy_u = geo.compute_spatial_steps(self.__lon_u, self.__lat_u)
        dx_v, dy_v = geo.compute_spatial_steps(self.__lon_v, self.__lat_v)
        
        J_u = np.sum((u + geo.compute_advection_v_jax(u, v, dx_v, dy_v)/self.__coriolis_factor_u.filled(1) - self.__u_geos.filled(0))**2)
        J_v = np.sum((v - geo.compute_advection_u_jax(u, v, dx_u, dy_u)/self.__coriolis_factor_v.filled(1) - self.__v_geos.filled(0))**2)

        return J_u + J_v
