"""Packages"""

from tkinter import *
from tkinter import filedialog
import xarray as xr
import os

def openFile(title):
    file = filedialog.askopenfilename(title = title, filetypes = (("NetCDF files", "*.nc*"), ("all files", "*.*")))
    ds = xr.open_dataset(file)
    
    return ds

print(os.path.dirname(__file__))