# Cyclogeostrophic balance code

This repository contains code for performing inversion of oceanographic data using various algorithms and approaches. The goal of the code is to estimate the velocity of an ocean system from sea surface height observations.

To use this code, you will need to install the following dependencies:

* JAX;
* Xarray.

## Usage

To use the inversion code, you will need to provide input data in the form of netCDF files, as well as a configuration files specifying the inversion parameters.

The inversion code implements algorithms for estimating the cyclogeostrophic velocity from the data, including:

* A new Gradient-based approach
* Iterative method

This repository also contains a file explaining the basic functions of the JAX library.