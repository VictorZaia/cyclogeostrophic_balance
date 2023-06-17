"""Packages"""

from tkinter import *
from tkinter import filedialog
import xarray as xr

from Model import *

"""Function to import file"""

def openFile(title):
    """
    Function that allows the user to interactively select a file
    Arguments:
    title - title of the file dialog;
    """
    file = filedialog.askopenfilename(title = title, filetypes = (("NetCDF files", "*.nc*"), ("all files", "*.*")))
    ds = xr.open_dataset(file)
    
    return ds

# =============================================================================
# Manual input of the path to the data
# =============================================================================
dir_data = '/home/victor/PyProject/data/'
name_mask = 'mask_eNATL60MEDWEST_3.6.nc'
name_ssh = 'eNATL60MEDWEST-BLB002_y2009m07d01.1h_sossheig.nc'
name_u = 'eNATL60MEDWEST-BLB002_y2009m07d01.1h_sozocrtx.nc'
name_v = 'eNATL60MEDWEST-BLB002_y2009m07d01.1h_somecrty.nc'


"""Class that will be responsible for reading the data and initializing the model"""

class PreProcessor:

    __model = Model() #static variable

    """Methods"""

    def read_SSH():
        """
        Function that reads the sea surface height file and stores the information in the model
        """
        try:
            # ds_ssh = openFile("Select the sea surface height file") # Automatically opens a file dialog
            ds_ssh = xr.open_dataset(dir_data+name_ssh) # Manual input of the path to the data
            PreProcessor.__model.set_lon_ssh(ds_ssh.nav_lon.values)
            PreProcessor.__model.set_lat_ssh(ds_ssh.nav_lat.values)
            PreProcessor.__model.set_ssh(ds_ssh.sossheig[0].values)
        except:
            print('Error: the ssh file could not be loaded. Check the path to the file (variable: dir_data) or if the name of the data to be imported are correct.')
        else:
            print('====SSH file read====')

    def read_u():
        """
        Function that reads the u velocity file and stores the information in the model
        """
        try:
            # ds_u = pre.openFile("Select the u velocity file") # Automatically opens a file dialog
            ds_u = xr.open_dataset(dir_data+name_u) # Manual input of the path to the data
            PreProcessor.__model.set_lon_u(ds_u.nav_lon.values)
            PreProcessor.__model.set_lat_u(ds_u.nav_lat.values)
            PreProcessor.__model.set_u(ds_u.sozocrtx[0].values)
        except:
            print('Error: the u component file could not be loaded. Check the path to the file (variable: dir_data) or if the name of the data to be imported are correct.')
        else:
            print('====U velocity file read====')

    def read_v():
        """
        Function that reads the v velocity file and stores the information in the model
        """
        try:
            # ds_v = pre.openFile("Select the v velocity file") # Automatically opens a file dialog
            ds_v = xr.open_dataset(dir_data+name_v) # Manual input of the path to the data
            PreProcessor.__model.set_lon_v(ds_v.nav_lon.values)
            PreProcessor.__model.set_lat_v(ds_v.nav_lat.values)
            PreProcessor.__model.set_v(ds_v.somecrty[0].values)
        except:
            print('Error: the v component file could not be loaded. Check the path to the file (variable: dir_data) or if the name of the data to be imported are correct.')
        else:
            print('====V velocity file read====')

    def read_mask():
        """
        Function that reads the mask file and stores the information in the model
        """
        try:
            # ds_mask = pre.openFile("Select the mask file") # Automatically opens a file dialog
            ds_mask = xr.open_dataset(dir_data+name_mask) # Manual input of the path to the data
            PreProcessor.__model.set_mask_ssh(ds_mask.tmask[0,0].values)
            PreProcessor.__model.set_mask_u(ds_mask.umask[0,0].values)
            PreProcessor.__model.set_mask_v(ds_mask.vmask[0,0].values)
        except:
            print('Error: the mask file could not be loaded. Check the path to the file (variable: dir_data) or if the name of the data to be imported are correct.')
        else:
            print('====Mask file read====')

    @staticmethod
    def initialize_model():
        """
        Function that initializes the model, calling all the functions that read the file and returns a model object containing all the data
        """
        PreProcessor.read_SSH()
        PreProcessor.read_u()
        PreProcessor.read_v()
        PreProcessor.read_mask()
        
        PreProcessor.__model.mask_data()
        PreProcessor.__model.compute_Coriolis_factor()

        return PreProcessor.__model
