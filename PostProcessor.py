"""Packages"""

import numpy as np
import xarray as xr

from PreProcessor import dir_data

"""Class that will be responsible writing the results to a netCDF file"""

class PostProcessor:

    @staticmethod
    def write_results(model):
        """
        Function that writes the results to a netCDF file
        Arguments:
        model - model that contains the data to be exported;
        """
        array_lat_u = np.asarray(model.get_lat_u())
        array_lat_v = np.asarray(model.get_lat_v())

        array_lon_u = np.asarray(model.get_lon_u())
        array_lon_v = np.asarray(model.get_lon_v())

        try:
            ds_u_out = xr.DataArray(data = model.get_u_cyclo(), dims = ['x', 'y'], coords = dict(lat = (['x', 'y'], array_lat_u), lon = (['x', 'y'], array_lon_u), mask_u = (['x', 'y'], model.get_mask_u())), name = 'Velocity_u')
            ds_v_out = xr.DataArray(data = model.get_v_cyclo(), dims = ['x', 'y'], coords = dict(lat = (['x', 'y'], array_lat_v), lon = (['x', 'y'], array_lon_v), mask_v = (['x', 'y'], model.get_mask_v())), name = 'Velocity_v')

            ds_u_out.to_netcdf(dir_data + 'u_cyclogeostrophic.nc')
            ds_v_out.to_netcdf(dir_data + 'v_cyclogeostrophic.nc')
        except:
            print('Error: the file could not be saved, check the variable dir_data and the variables that are being exported.')
        else:
            print('====Results exported====')
