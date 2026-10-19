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
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

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

from small_island_functions import adjust_lon_coords
from small_island_functions import island_bounds_box

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


############# getting MHW data ############# f"{root_noaa}", f"{var_name}_ds.nc")

var_name = f"{var_name}_mhw"
print('variable', {var_name})

# Define expected patterns explicitly
globals()[var_name] = xr.open_dataset(
        os.path.join(
            f"{root_noaa}", f"{var_name}_ds.nc")
        )



################################ load historical ClimateDT data ###############################################

var = "avg_tos"
var2 = "tos"

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

    

    ##### now getting MHW data ############## 

    # Define expected patterns explicitly
    pattern = f"{names[index]}_{var2}_regrid_mhw_ds.nc"

    all_files = os.listdir(folder_path)

    files = [
        os.path.join(folder_path, filename)
        for filename in all_files
        if fnmatch.fnmatch(filename, pattern)
    ]

    ds = xr.open_dataset(files[0])

    var_name = f"{names[index]}_tos_mhw"
    print('variable', {var_name})
    # adjusting to degree C 
    globals()[var_name] = ds 


start_time = "1992-01-01"
end_time = "2018-12-31"

regrid = True 

percentile = 90

######################## calculating percentile threshold ##################### 

exps = ["noaa_tos", "nemo_tos", "fesom_tos", "icon_tos"] # , "mean_bias"]

lat_bounds = [19,8] # 20, 7
lon_bounds = [-70, -55] # -80 to -50


for index, exp in enumerate(exps):

    print(exp)
    # formatting to location and time mean
    globals()[f"{exp}"] = globals()[f"{exp}"].sel(
                lat = slice(lat_bounds[0], lat_bounds[1]), 
                lon = slice(lon_bounds[0], lon_bounds[1]),
            )
    
    # print(exp)
    da_mean = globals()[f'{exp}']

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





######################### plotting ###########################

fsize = 12 

exp_name = "nemo"

exp_names = ["noaa"] # 

projection = ccrs.PlateCarree()

for exp_name in exp_names:
    cmap = cmo.cm.haline
    cmap = ListedColormap(cmap(np.linspace(0, 1, 20)))

    cmap1 = cmo.cm.thermal
    cmap1 = ListedColormap(cmap1(np.linspace(0, 1, 20)))
    cmaps = [cmap1, cmap]

    climits = [[0, 2], [0, 100]]
    exps = ["sst_mhw", "event_duration"]  # , "event_max_area", "event_id"]
    cmap_labels = ["SST above 90th\n percentile threshold (C)", "duration (days)"]

    # --- Data, sliced once (load into memory so each frame is fast) ---
    mhw = globals()[f"{exp_name}_tos_mhw"].sel(time=slice(start_time, None))
    threshold = globals()[f"{exp_name}_tos_threshold_doy"].sel(time=slice(start_time, None))
    mhw = mhw.load()
    threshold = threshold.load()

    event_max = mhw["event_id"].max(dim=["lat", "lon"], skipna=True).values.astype(float)
    # treat "no event" (0 or NaN) as missing, carry the last event id forward
    event_max[~(event_max > 0)] = np.nan
    event_held = pd.Series(event_max).ffill().fillna(0).to_numpy()

    def get_frame_data(exp, t):
        """Return the 2D field for variable `exp` at time index `t`."""
        data = mhw[exp].isel(time=t)
        if exp == "sst_mhw":
            data = data - threshold.isel(time=t)
        return data


    # --- Time indices to animate (indices into the sliced data) ---
    # frame_indices = range(1900, 1900 + 60)  # or 
    frame_indices = range(mhw.sizes["time"]) # for everything

    # --- Figure: built once ---
    fig, axes = plt.subplots(
        int(len(exps) / 2), 2,
        figsize=(10, 3.6 * len(exps) / 2),
        subplot_kw={"projection": projection},
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()

    first = frame_indices[0]
    meshes = []

    for index, (exp, ax_map) in enumerate(zip(exps, axes)):
        to_plot = get_frame_data(exp, first)

        pcm = ax_map.pcolormesh(
            to_plot.lon, to_plot.lat, to_plot,
            transform=projection,
            cmap=cmaps[index], shading="auto",
            vmin=climits[index][0], vmax=climits[index][1],
        )
        meshes.append(pcm)

        ax_map.set_extent(
            [
                float(to_plot.lon.min()), float(to_plot.lon.max()),
                float(to_plot.lat.min()), float(to_plot.lat.max()),
            ],
            crs=projection,
        )
        ax_map.set_aspect("auto")
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.set_title(exp, fontsize=fsize)

        gl = ax_map.gridlines(draw_labels=False, linestyle="dashed", alpha=0.5)
        if index in [0, 2]:
            gl.left_labels = True
            gl.bottom_labels = True

        cbar = fig.colorbar(pcm, ax=ax_map, orientation="vertical", pad=0.05, extend="max")
        cbar.set_label(cmap_labels[index], fontsize=fsize)

    title = fig.suptitle("", fontsize=fsize)


    def update(frame):
        t = frame_indices[frame]
        for exp, pcm in zip(exps, meshes):
            data = get_frame_data(exp, t)
            pcm.set_array(np.asarray(data).ravel())

        date_str = pd.Timestamp(mhw.time.values[t]).strftime("%Y-%m-%d")
        event_number = event_max[t]
        title.set_text(
            f"Marine heatwaves in {exp_name}\n"
            f"date {date_str}\n"
            f"event number {int(event_number) if np.isfinite(event_number) else 'none'}"
        )
        return [*meshes, title]


    file_path = f"/scratch/b/b382291/gifs/mhw_{exp_name}.mp4"
    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=100, blit=False)


    writer = FFMpegWriter(
        fps=10, codec="libx264", bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )

    # writer = FFMpegWriter(fps=10, codec="libx264", bitrate=-1,
    #                     extra_args=["-pix_fmt", "yuv420p"])
    animation.save(file_path, writer=writer, dpi=150)


    #animation.save(file_path, writer="pillow", fps=10)
    # For an mp4 instead (needs ffmpeg, usually much smaller than a gif):
    #animation.save(file_path, writer="ffmpeg", fps=10, dpi=150)
