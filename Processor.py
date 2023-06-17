"""Packages"""

import numpy as np
import numpy.ma as ma

import Tools.geometry as geo
import Tools.geostrophy as geos
import Tools.cyclogeostrophy as cyclo

"""Class that will be responsible for solving the cyclogeostrophic balance"""

class Processor:

    def compute_geostrophy(model):
        """
        Funtion that computes the geostrophic velocities from the data contained in the object model, needed to solve the cyclogeostrophic balance
        Arguments:
        model - model that contains the information about the sea surface height and the coriolis factor;
        """
        dx_ssh, dy_ssh = geo.compute_spatial_steps(model.get_lon_ssh(), model.get_lat_ssh())
        u_geos, v_geos = geos.geostrophy(model.get_ssh(), dx_ssh, dy_ssh, model.get_coriolis_factor_u(), model.get_coriolis_factor_v())

        u_geos = ma.masked_array(u_geos, model.get_mask_u())
        v_geos = ma.masked_array(v_geos, model.get_mask_v())

        model.set_u_geos(u_geos)
        model.set_v_geos(v_geos)

    def compute_cyclogeostrophy_variational(f,model):
        """
        Funtion that computes the cyclogeostrophic velocities from the data contained in the object model.
        This function uses the variational formulation approach to solve the cyclogeostrophic balance.
        Arguments:
        model - model that contains the data needed;
        """
        u = np.copy(model.get_u_geos().filled(0))
        v = np.copy(model.get_v_geos().filled(0))

        dx_ssh, dy_ssh = geo.compute_spatial_steps(model.get_lon_ssh(), model.get_lat_ssh())

        u_min, v_min = cyclo.gradient_descent_jax(f,u ,v)

        u_min = ma.masked_array(u_min, model.get_mask_u())
        v_min = ma.masked_array(v_min, model.get_mask_v())
    
        model.set_u_cyclo(u_min)
        model.set_v_cyclo(v_min)
    
    def compute_cyclogeostrophy_iterative(model):
        """
        Funtion that computes the cyclogeostrophic velocities from the data contained in the object model.
        This function uses the iterative approach to solve the cyclogeostrophic balance.
        Arguments:
        model - model that contains the data needed;
        """
        u = np.copy(model.get_u_geos().filled(0))
        v = np.copy(model.get_v_geos().filled(0))

        u_it, v_it = cyclo.cyclogeostrophy(u, v, model.get_coriolis_factor_u(),model.get_coriolis_factor_v(), model.get_lon_u(), model.get_lat_u(), model.get_lon_v(), model.get_lat_v(), 0.0001)
        
        model.set_u_cyclo(u_it)
        model.set_v_cyclo(v_it)

    @staticmethod
    def solve_model(model, method = 0):
        """
        Function that solves the cyclogeostrophic balance from the initialized model.
        Both the variational formulation and the iterative method are available.
        Arguments:
        model - model that contains the data needed;
        method - method used to solve the cyclogeostrophic balance. 0 for variational formulation and 1 for iterative method.
        """
        print('====Solving started====')

        Processor.compute_geostrophy(model)

        if method == 0:
            Processor.compute_cyclogeostrophy_variational(model.loss, model)
            print('====Solving finished====')
        elif method == 1:
            Processor.compute_cyclogeostrophy_iterative(model)
            print('====Solving finished====')
        else:
            print('Error: invalid method. Chose 0 for variational formulation or 1 for iterative method.')
