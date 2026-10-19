# run the marine heatwaves code on the different models at both resolutoins 

import os 

import xarray as xr
import numpy as np
import pandas as pd 
import fnmatch
from numpy.polynomial import Polynomial
import seaborn as sns

import cartopy.feature as cf
import healpy as hp
import cftime
from datetime import datetime, timedelta
import pyproj
import regionmask
import scipy.stats as sstats
from scipy.interpolate import griddata
import xesmf as xe 
import cmocean as cmo 
import matplotlib.ticker as mticker

#import easygems.healpix as egh
#from scipy.spatial import KDTree
import matplotlib.tri as tri

import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.gridspec as gridspec

import numpy as np
from matplotlib.lines import Line2D



import xclim
from xclim.core.calendar import percentile_doy

# import aqua
# from aqua import Reader
# from aqua import inspect_catalog
import pyproj
pyproj.datadir.set_data_dir("/home/bsc/bsc032263/.conda/envs/aqua/share/proj")

os.chdir("/home/b/b382291/git/small_islands")

from small_island_functions import interpolate_timeseries
from small_island_functions import adjust_lon_coords
from small_island_functions import island_bounds_box
from small_island_functions import format_data
from small_island_functions import matching_resolutions_xe
from small_island_functions import summarize_mhw_events
from small_island_functions import detect_mhw_events
# from small_island_functions import summarize_period


island_bounds = island_bounds_box()

###################33 PRECIPITATION #####################################
#### import data from DestinE and from CORDEX

diff = 273.15 

root_noaa =  "/work/bb1153/b382291/caribbean/noaa_oisstv2"
root_esa =  "/work/bb1153/b382291/caribbean/esa_sstcci"
root_ifs_nemo = "/work/bb1153/b382291/caribbean/IFS-NEMO/0.1"
root_ifs_fesom = "/work/bb1153/b382291/caribbean/IFS-FESOM/0.1"
root_icon = "/work/bb1153/b382291/caribbean/ICON/0.1"

# root_mswep = "/gpfs/scratch/bsc32/bsc032263/mswep"
# root_era5_land = "/gpfs/scratch/bsc32/bsc032263/era5land"
# root_cimh = "/gpfs/scratch/bsc32/bsc032263/cimh"

islands_all = ["caribbean_islands"] 
islands_all_short = ["car"] 

var = "tos"

###############################3 load observational data ###############################################
noaa_sst = xr.open_dataset(f"{root_noaa}/tos_1990_01_01_to_2019_03_31.nc")
noaa_sst = noaa_sst[f"sst"]

# want latitude to run from north to south
noaa_sst = noaa_sst.sortby("lat", ascending=False)
noaa_sst = adjust_lon_coords(noaa_sst)

# print and name variable 
var_name = f"noaa_{var}"
print('variable', {var_name})
globals()[var_name] = noaa_sst.drop_duplicates("time")

###############################3 load observational data ###############################################

esa_sst = xr.open_dataset(f"{root_esa}/tos_1990_01_to_2019_12_monthly_mean.nc")
esa_sst = esa_sst[f"{var}"]

# want latitude to run from north to south
esa_sst = esa_sst.sortby("lat", ascending=False)

# print and name variable 
var_name = f"esa_{var}"
print('variable', {var_name})
globals()[var_name] = esa_sst.drop_duplicates("time")


################################ load historical ClimateDT data ###############################################


percentile = 90

regrid = True
var = "avg_tos"

# CURRENTLY REMOVING ICON AS MISSING 2014 
roots = [root_ifs_nemo, root_ifs_fesom, root_icon]
names = ["nemo", "fesom", "icon"] # 

for index, root in enumerate(roots):

    folder_path = f"{root}"
    # Define expected patterns explicitly
    pattern_1990 = f"{var}_1990*car_crop.nc"
    pattern_2015 = f"{var}_2015*car_crop.nc"

    all_files = os.listdir(folder_path)

    files_1990 = [
        os.path.join(folder_path, filename)
        for filename in all_files
        if fnmatch.fnmatch(filename, pattern_1990)
    ]
    files_2015 = [
        os.path.join(folder_path, filename)
        for filename in all_files
        if fnmatch.fnmatch(filename, pattern_2015)
    ]

    ds1 = xr.open_dataset(files_1990[0])
    ds2 = xr.open_dataset(files_2015[0])

    ds = xr.concat([ds1, ds2], "time")

    #rename to lat lon
    ds = ds.rename({'latitude':'lat'})
    ds = ds.rename({'longitude':'lon'})

    # want to use the - notatin
    ds = adjust_lon_coords(ds[f"{var}"])
    
    var_name = f"{names[index]}_tos"
    print('variable', {var_name})
    # adjusting to degree C 
    globals()[var_name] = ds - diff 


############### calculate MHW ################## 

exps = ["noaa_tos", "nemo_tos", "fesom_tos", "icon_tos"] # , "mean_bias"]

lat_bounds = [19,8] # 20, 7
lon_bounds = [-70, -55] # -80 to -50

start_time = "1992-01-01"
end_time = "2018-12-31"

for index, exp in enumerate(exps):

    print(exp)
    # formatting to location and time mean
    globals()[f"{exp}"] = globals()[f"{exp}"].sel(
                lat = slice(lat_bounds[0], lat_bounds[1]), 
                lon = slice(lon_bounds[0], lon_bounds[1]),
            )
    
    # print(exp)
    da_mean = globals()[f'{exp}']

    # # # Collapse to domain-averaged SST (1D time series)
    # # # adjust dim names ('lat','lon') to match your dataset
    # # da_mean = da #.mean(dim=["lat", "lon"], skipna=True)

    if regrid and exp in ["nemo_tos", "fesom_tos", "icon_tos"]:

        # regrid grid in, grid out 
        regridder = xe.Regridder(da_mean ,globals()[f"noaa_tos"], "bilinear")
        da_mean = regridder(da_mean, keep_attrs=True)

        # da_mean = matching_resolutions_xe(
        #     da_mean, globals()[f"noaa_tos"], False,  method = "bilinear"
        # )

        globals()[f'{exp}'] = da_mean
    

    # the length of time you want to make your MHW threshold 
    da_mean_clim = da_mean.sel(time = slice(start_time, end_time))

    # Compute day-of-year 90th percentile threshold with 11-day window
    threshold_doy = percentile_doy(
        da_mean_clim,
        window=11,
        per=percentile
    )

    if "percentiles" in threshold_doy.coords:
        threshold_doy = threshold_doy.squeeze("percentiles", drop=True)

    # ignoring the first 2 years 
    to_plot = da_mean.sel(time = slice(start_time, None))
    threshold_full = threshold_doy.sel(dayofyear=to_plot.time.dt.dayofyear)
    globals()[f"{exp}_threshold_doy"] = threshold_full


    # detect MHW over the whole period 
    #ds = globals()[f"{exp}"]


    print(f'starting MHW detection for {exp}')
    globals()[f"{exp}_mhw_ds"] = detect_mhw_events(
                                    da_mean.sel(time = slice(start_time, None)), 
                                    globals()[f"{exp}_threshold_doy"], 
                                    min_duration=5
                                )

    globals()[f"{exp}_mhw_summary"]  = summarize_mhw_events(
                                    globals()[f"{exp}_mhw_ds"]
                                )

    if regrid and exp in ["nemo_tos", "fesom_tos", "icon_tos"]:

        name = f"{exp}_regrid"
    else:
        name = exp

    file_path = os.path.join("/scratch/b/b382291/carribean", f"{name}_mhw_ds.nc")
    globals()[f"{exp}_mhw_ds"].to_netcdf(path = file_path, mode = 'w')

    # file_path2 = os.path.join("/scratch/b/b382291/carribean", f"{name}_mhw_summary")
    # globals()[f"{exp}_mhw_summary"].to_netcdf(path = file_path2, mode = 'w')