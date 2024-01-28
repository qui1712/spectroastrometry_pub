"""
Created on Tue Jul 18 16:56:40 2023.

@author: Quirijn B. van Woerkom
Code to produce the plots accompanying the spectroastrometry paper.
"""
# Standard imports
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
import time
import tqdm

# Import spectroastrometry_tools
import spectroastrometry_tools as st
# and spectroastrometry_orbit
from spectroastrometry_orbit import Orbit, xi_hat, solve_Kepler
# and filters
from spectroastrometry_filters import filters



# Plotstyle changes
# Increase the matplotlib font size
plt.rcParams.update({"font.size": 22})
# Set layout to tight
plt.rcParams.update({"figure.autolayout": True})
# Set grid to true
plt.rcParams.update({"axes.grid": False})
# Set the plotting style to interactive (autoshows plots)
plt.ion()

# Close everything
plt.close('all')

# %% Initialise instrument options
# ############# ELT
telescope = "ELT"
instrument = "METIS"
# Set the achromatic efficiency component of the telescope
eps_ELT_throughput = .36  # From Carlomagno et al. (2020): total system
# throughput for ELT and METIS.
eps_atmos_trans = .8  # Atmospheric transmission: ballpark figure from
# ESO SkyCalc
eps = eps_ELT_throughput*eps_atmos_trans  # Product of the two
# Set the observation time
t = 6*u.h
# Preset the planet filter
planet_filter = "M'"
moon_filter = "N2"
# Print the PSF standard deviations
print(filters.filters[planet_filter].PSF_sigma)
print(filters.filters[moon_filter].PSF_sigma)

# %% Create system, star and planet
###############################################################################
# eps Indi A(b)
# Distance
mps = st.MoonPlanetSystem(dist=3.6481*u.pc)
# Create eps Indi Ab
mps.create_gas_giant_from_grid(
    age=3.0*u.Gyr,  # Max that ATMO2020 can do
    mass=3.25*u.Mjup,
    # mass=1*u.Mjup,
    name="$\\epsilon$ Indi Ab",
    equil_chem="CEQ"
)
# Create eps Indi A
mps.Star.set_blackbody(
    Teff=4560*u.K,
    radius=.732*u.Rsun,
    wavs=mps.Planet.wavs,
    name="$\\epsilon$ Indi A"
)
mps.Star.mass = .762*u.Msun
# Set the orbit of eps Indi Ab
mps.Planet.orbit = Orbit(
    a=11.55*u.AU,
    e=.26,
    i=64.25*u.deg,
    Omg=(250.20+90)*u.deg,
    omg=77.83*u.deg,
    Mref=143.8*u.deg,
    M0=mps.Star.mass
)
###############################################################################
# %% Create the moon
# ####################### Blackbody moon ######################################
moon_Teff = 350*u.K
moon_mass = 1*u.Mearth
moon_radius = 1.0*u.Rearth
radius_frac = 1.0
moon_name = (f"{moon_Teff.value} K, {radius_frac} "
              "$ R_{\\oplus}$ moon")
mps.create_blackbody_moon(Teff=moon_Teff,
                          radius=moon_radius,
                          wavs=mps.Planet.wavs,
                          name=moon_name
                          )
mps.Moons[0].mass = moon_mass

# %% Produce spectrum graph
mps.plot_spectra(include_star=False, include_tot=False, contrast=None,
                 spectral_unit=u.ph/u.s/u.um/u.m**2,
                 # spectral_unit=u.W/u.cm**2/u.sr/u.um,
                 filter_plot=[planet_filter, moon_filter],
                 wavelength_window=[2.5, 15])

# %% Produce a-min_T graph
r_moon_arr = np.array([.286, .53, 1.])*u.Rearth
# r_moon_arr = np.array([.286, 1.])*u.Rearth
mps.plot_min_T_R(instrument, t, eps, planet_filter, moon_filter,
                 r_moon_arr=r_moon_arr, nvals_a=120,
                 plot_tides=False)
# %% Produce Tmin(e, t0P)-graph
# Set moon radii
r_moons = np.array([.286, 1.])*u.Rearth
# Set moon semi-major axes
a_moons = np.array([421.7e3, 1882.7e3])*u.km
# Set minimum and maximum temperatures to consider
min_T = np.array([250, 140])
max_T = np.array([310, 170])
n_vals = 50
mps.min_bb_temp_e_t0P(instrument, t, eps, planet_filter, moon_filter, r_moons,
                      a_moons, nvals_e=n_vals, nvals_t0P=n_vals, min_Ts=min_T,
                      max_Ts=max_T)

# %% Produce SNR(T, d)-graph
mps.plot_SNR_T_d(instrument, t, eps, planet_filter, moon_filter, r_moons,
                  a_moons, n_T=300, n_d=300)
