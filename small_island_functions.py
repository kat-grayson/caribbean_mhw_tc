# small island functions
from __future__ import annotations


import xarray as xr
import numpy as np
import pandas as pd 
import seaborn as sns
import smmregrid
from smmregrid import Regridder, CdoGenerate
from numpy.polynomial import Polynomial
import xesmf as xe 

import healpy as hp
import cmocean
import pyproj
import regionmask
from scipy.interpolate import griddata

#import easygems.healpix as egh
import matplotlib.tri as tri

import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

from scipy import ndimage

import pyproj
pyproj.datadir.set_data_dir("/home/bsc/bsc032263/.conda/envs/aqua/share/proj")



def convert_mtpr_tp(input_data):
    # convert from kg / m2 s to mm / hr
    # density of water = 1000 kg / m3 
    factor = 0.2777777777777778
    input_data = input_data/factor
    input_data =  1000*input_data
    
    return input_data

def convert_prlr_tp(input_data):
    # convert from m / s to mm/hr
    input_data =  (input_data) * 1000  * 60 * 60
    return input_data


# defining a function to give the closest lat lon indicies based on a target lat lon
def closest_value(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]




# def interpolate_timeseries(ref_lat, ref_lon, data, method = "nearest"):
#     """Takes a reference lat/lon from an observational station and uses
#     cubic interpolation to create a time series from the gridded model
#     data"""
#     # Extract lats and lons from the data array
    
#     try:
#         data['lon_adjusted'] = xr.where(data.lon > 180, data.lon - 360, data.lon)
#         data['lon'] = data['lon_adjusted']
#         data = data.reset_coords(names='lon_adjusted', drop=True)

#         lats = data['lat'].values
#         lons = data['lon'].values
        
#     except AttributeError:
#         data['lon_adjusted'] = xr.where(data.rlon > 180, data.rlon - 360, data.rlon)
#         data['rlon'] = data['lon_adjusted']
#         data = data.reset_coords(names='lon_adjusted', drop=True)

#         lats = data['rlat'].values
#         lons = data['rlon'].values

#     #print('ref lon', ref_lon, "ref_lat", ref_lat)

#     target_lon_nn = closest_value(lons, ref_lon)
#     target_index_lon = np.where(lons == target_lon_nn)[0][0]
#     target_lat_nn = closest_value(lats, ref_lat)
#     target_index_lat = np.where(lats == target_lat_nn)[0][0]

#     #print('target lon', target_lon_nn, "target_lat", target_lat_nn)
    
#     # time series based on nearest neighbour
#     try:
#         ts =  data[:,target_index_lat, target_index_lon]
#     except IndexError:
#         ts =  data[target_index_lat, target_index_lon]

#     if method == "nearest":

#         return ts
    
#     else:

#         station_ts = xr.zeros_like(ts)

#         points = np.array([(lat, lon) for lat in lats for lon in lons])

#         # doing the interpolation on each time step
#         try:
#             length = len(data.time)
#             for t in tqdm.tqdm(range(length)):
#                 values = data.values[t,:]  # Your 2D climate data
#                 # Flatten lat, lon, and value arrays for interpolation
#                 values_flat = values.flatten()
#                 station_ts.data[t] = griddata(points, values_flat, (ref_lat, ref_lon), method=method)
#         except AttributeError:
#             values = data.values  # Your 2D climate data
#             # Flatten lat, lon, and value arrays for interpolation
#             values_flat = values.flatten()
#             ts = griddata(points, values_flat, (ref_lat, ref_lon), method=method)
#             station_ts.data = ts[0]


#         return station_ts


def interpolate_timeseries(ref_lat, ref_lon, data, method = "nearest"):
    """Takes a reference lat/lon from an observational station and uses
    cubic interpolation to create a time series from the gridded model
    data"""
    # Extract lats and lons from the data array

    try:
        data['lon_adjusted'] = xr.where(data.lon > 180, data.lon - 360, data.lon)
        data['lon'] = data['lon_adjusted']
        data = data.reset_coords(names='lon_adjusted', drop=True)

    except AttributeError:
        data['lon_adjusted'] = xr.where(data.rlon > 180, data.rlon - 360, data.rlon)
        data['rlon'] = data['lon_adjusted']
        data = data.reset_coords(names='lon_adjusted', drop=True)

    if method == "nearest":
        try:
            ts = data.sel(lat = ref_lat, lon = ref_lon, method = 'nearest')
            #ts =  data[:,target_index_lat, target_index_lon]
        except AttributeError:
            ts = data.sel(rlat = ref_lat, rlon = ref_lon, method = 'nearest')
            #ts =  data[target_index_lat, target_index_lon]

        return ts
    
    else:

        station_ts = data.interp(lat = ref_lat, lon = ref_lon, method = method)

        if np.isnan(station_ts).any():
            station_ts = data.interp(lat = ref_lat, lon = ref_lon, method = "nearest")

        return station_ts


def select_box_area(data, latitude, longitude):
    
    #coordinates of Canary islands whole domain 
    latitude = [27.2, 29.5]
    longitude = [-19, -12.5]

    # converting co-ords into degrees from radian
    # data['lon'] = data['lon']*180/(np.pi)
    # data['lat'] = data['lat']*180/(np.pi)

    data['lon_adjusted'] = xr.where(data.lon > 180, data.lon - 360, data.lon)
    data['lon'] = data['lon_adjusted']
    data = data.reset_coords(names='lon_adjusted', drop=True)
    
    indexer = ((data.lat>=latitude[0]) & (data.lat<=latitude[1]) & (data.lon>=longitude[0]) & (data.lon<=longitude[1])).compute()
    data = data.where(indexer, drop=True)

    return data


def create_land_mask(input_data):
    
    "creates land mask for countries of interest"
    "[52, 308, 780] is Caribbean"
    country_codes = [37, 47, 54, 78, 103, 121, 122, 137, 156, 160, 224, 234, 235, 52, 308, 780]

    try:
        lat_land = input_data.lat.values 
        lon_land = input_data.lon.values
    except AttributeError:
        lat_land = input_data.rlat.values 
        lon_land = input_data.rlon.values
        
    #The mask has different flag values for each country and nans for the ocean
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_50
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_50

    mask_land = land.mask(lon_land, lat_land)
    mask_land
    mask_countries = countries.mask(lon_land, lat_land)
    mask_countries = mask_countries.astype(int)

    output_data = input_data.where(np.isin(mask_countries, country_codes), np.nan)
    
    return output_data

def create_island_mask(input_data):
    
    country_codes = [54, 160, 52, 308, 780, 4, 38,  71, 72, 219] # SPain 54, france 160, 
    # caribbean  4, 38,  71, 72, 219
    try:
        lat_land = input_data.lat.values 
        lon_land = input_data.lon.values
    except AttributeError:
        lat_land = input_data.rlat.values 
        lon_land = input_data.rlon.values
        

    #The mask has different flag values for each country and nans for the ocean
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_50
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_50

    mask_land = land.mask(lon_land, lat_land)
    mask_land
    mask_countries = countries.mask(lon_land, lat_land)
    mask_countries = mask_countries.astype(int)

    output_data = input_data.where(np.isin(mask_countries, country_codes), np.nan)
    
    return output_data


def create_ocean_mask(input_data):

    # 1. Load Natural Earth ocean basins
    basins = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50

    try:
        lat_ocean = input_data.lat.values
        lon_ocean = input_data.lon.values
    except AttributeError:
        lat_ocean = input_data.rlat.values
        lon_ocean = input_data.rlon.values

    # The mask has different flag values for each ocean basin and NaN over land
    mask_basins = basins.mask(lon_ocean, lat_ocean)

    output_data = input_data.where(~np.isnan(mask_basins.values), np.nan)

    return output_data


def ticks_round(ticks: list, round_to: int = None):
    """
    Round a tick to the nearest round_to value.

    tick (list):          Tick value.
    round_to (int, opt):  Round to value.

    Returns:
        tick (list):  Rounded tick value.
    """
    if round_to is None:
        # define round_to
        tick_span = ticks[1] - ticks[0]
        if tick_span <= 1:
            round_to = 2
        elif tick_span > 1 and tick_span <= 10:
            round_to = 1
        else:
            round_to = 0

    return np.round(ticks, round_to)

def plot_islands(to_plot, grid_type, min_value, max_value, latitude, longitude, title, cbar_label, diff_or_cont = "diff"):

    projection = ccrs.PlateCarree()

    fsize = 14
    #Add tick marks with values for latitude and longitude axes
    nxticks = 5
    nyticks = 7
    
    # coordinates of La Palma southern tip 
    # this part was to see where the same 'real' co-ordinate was plotted on different re-grids
    # lat_palma = 28.455732
    # lon_palma = -17.838980

    # lat_cruz = 28.100639
    # lon_cruz = -14.499162

    # if grid_type == 'lat_lon':
    #     closest_point = to_plot.sel(lat=lat_palma, lon=lon_palma, method='nearest')
    #     loni = np.where(to_plot.lon == closest_point.lon.values)[0][0]
    #     lati = np.where(to_plot.lat == closest_point.lat.values)[0][0]
    #     print('closest point palma lat:', closest_point.lat.values, 'lon:', closest_point.lon.values,
    #         'real: ', lat_palma, lon_palma)

    #     closest_point = to_plot.sel(lat=lat_cruz, lon=lon_cruz, method='nearest')
    #     loni2 = np.where(to_plot.lon == closest_point.lon.values)[0][0]
    #     lati2 = np.where(to_plot.lat == closest_point.lat.values)[0][0]
    #     print('closest point cruz lat:', closest_point.lat.values, 'lon:', closest_point.lon.values,
    #         'real: ', lat_cruz, lon_cruz)

    if diff_or_cont == "cont":
        cmap = cmocean.cm.thermal
        cmap = cmap(np.linspace(0, 1, 20))
        cmap = ListedColormap(cmap)
    elif diff_or_cont == "diff":
        cmap = plt.get_cmap('RdBu_r', 20)
        
    fig, ax = plt.subplots(1, 1, constrained_layout=False, figsize=(10, 10), 
                            subplot_kw={'projection': projection}) 


    if grid_type == "lat_lon":

        try:
            lons, lats = np.meshgrid(to_plot.rlon.values, to_plot.rlat.values)
        except AttributeError:    
            lons, lats = np.meshgrid(to_plot.lon.values, to_plot.lat.values)
        
        pcm = ax.pcolormesh(lons, lats, to_plot, transform= projection, 
            cmap=cmap, vmax = max_value, vmin = min_value) 
        
        # ax.plot(to_plot.lon[loni], to_plot.lat[lati], marker='x', color='red', markersize=6, transform=projection)
        # ax.plot(to_plot.lon[loni2], to_plot.lat[lati2], marker='x', color='red', markersize=6, transform=projection)
    
    # elif grid_type == "healpix":
        
    #     ax.set_global()
    #     ax.set_extent([longitude[0], longitude[1], latitude[0], latitude[1]], crs=projection)
    #     pcm = egh.healpix_show(to_plot, ax=ax, vmax = max_value, cmap = cmap)

    elif grid_type == "native":
        triang = tri.Triangulation(to_plot.lon, to_plot.lat)
        pcm = plt.tricontourf(triang, to_plot.data, alpha=1, levels=np.linspace(0, max_value, 20),
            cmap=cmap, transform= projection) 

    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        
    plt.title(f'{title}')

    # Latitude labels
    if grid_type == "healpix":
        lat_min = latitude[0]
        lat_max = latitude[1]
    else:
        try:
            lat_min = to_plot['lat'].values.min()
            lat_max = to_plot['lat'].values.max()
        except KeyError:
            lat_min = to_plot['rlat'].values.min()
            lat_max = to_plot['rlat'].values.max()
            
    step = (lat_max - lat_min) / (nyticks - 1)
    yticks = np.arange(lat_min, lat_max, step)
    yticks = ticks_round(ticks=yticks, round_to=None)
    ax.set_yticks(yticks, crs=projection)
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

    # Latitude labels
    if grid_type == "healpix":
        lon_min = longitude[0]
        lon_max = longitude[1]
    else:
        try:
            lon_min = to_plot['lon'].values.min()
            lon_max = to_plot['lon'].values.max()
        except KeyError:
            lon_min = to_plot['rlon'].values.min()
            lon_max = to_plot['rlon'].values.max()
    step = (lon_max - lon_min) / (nxticks - 1)
    xticks = np.arange(lon_min, lon_max + 1, step)
    xticks = ticks_round(ticks=xticks, round_to=None)
    ax.set_xticks(xticks, crs=projection)
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

    ax.tick_params(axis='y', labelsize=fsize) 
    ax.tick_params(axis='x', labelsize=fsize) 

    try:
        to_plot['lon'].values.max()
        cbar_ax = fig.add_axes([0.36, 0.3, 0.28, 0.01]) 
    except KeyError:
        cbar_ax = fig.add_axes([0.36, 0.27, 0.28, 0.01]) 
        
    cbar = fig.colorbar(pcm, cax = cbar_ax, orientation='horizontal', shrink=0.4)
    
    cbar.set_ticks([np.round(min_value,2),
                    np.round(min_value + (max_value - min_value)/3,2), 
                    np.round(min_value + 2*(max_value - min_value)/3,2), 
                    np.round(max_value,2)])

    cbar.update_ticks()  # Updates the colorbar with the new ticks

    cbar.ax.tick_params(labelsize=fsize)
    cbar.set_label(f"{cbar_label}", fontsize = fsize)

    plt.show()
    
    



def adjust_lat_coords(data):

    try:
        data['lon_adjusted'] = xr.where(data.lon > 180, data.lon - 360, data.lon)
        data['lon'] = data['lon_adjusted']
        data = data.reset_coords(names='lon_adjusted', drop=True)
    except AttributeError:
        data['lon_adjusted'] = xr.where(data.longitude > 180, data.longitude - 360, data.longitude)
        data['lon'] = data['lon_adjusted']
        data = data.reset_coords(names='lon_adjusted', drop=True)
    
    return data


def adjust_lon_coords(data):

    try:
        data['lon_adjusted'] = xr.where(data.lon > 180, data.lon - 360, data.lon)
        data['lon'] = data['lon_adjusted']
        data = data.reset_coords(names='lon_adjusted', drop=True)
    except AttributeError:
        data['lon_adjusted'] = xr.where(data.longitude > 180, data.longitude - 360, data.longitude)
        data['lon'] = data['lon_adjusted']
        data = data.reset_coords(names='lon_adjusted', drop=True)
    
    return data



def matching_resolutions_xe(grid_in, grid_out, method = "conservative_normed"):
    """Function for re-gridding the higher resolution observational
    grid to the lower resolution model grid. Based on 
    https://pangeo-xesmf.readthedocs.io/en/latest/notebooks/Masking.html
    
    Inputs:
    grid_in: xr.Dataarray. Model data with land sea mask already applied.
    grid you want to grid onto. 
    ESSENTIAL THAT MASK ALREADY APPLIED BEFORE INPUT.
    
    grid_out: xr.Dataarray. Grid that you want the final result to be onto.
    Land sea mask built
    into the data as only land obs based off station data.
    
    Ouptut:
    Interpolated values: xr.Dataarray 
    If grid in = observational data and grid out = model data 
    this will move the obs data ont on the same grid as the low_res model data. 
    """

    # adding mask
    grid_in_ds = grid_in.to_dataset(name = "zeta")
    grid_in_ds["mask"] = xr.where(~np.isnan(grid_in_ds["zeta"]), 1, 0)
    
    #converting high res obs into ds and adding mask
    grid_out_ds = grid_out.to_dataset(name = "zeta")
    grid_out_ds["mask"] = xr.where(~np.isnan(grid_out_ds["zeta"]), 1, 0)

    # regrid 
    regridder = xe.Regridder(grid_in_ds, grid_out_ds, method)
    interpolated_values = regridder(grid_in_ds["zeta"], keep_attrs=True)
    

    return interpolated_values 


def matching_resolutions_xe_temp_corr(low_res, high_res, low_res_orog, high_res_orog, method = "conservative_normed"):
    """
    Regrid high-res temperature to the low-res grid and apply a temperature height correction.

    Parameters:
        low_res (xarray.DataArray or Dataset): Low-res target grid with 'lat' and 'lon'.
        high_res (xarray.DataArray or Dataset): High-res temperature field.
        high_res_orog (xarray.DataArray): Orography (elevation) of high-res data (same grid as high_res).
        low_res_orog (xarray.DataArray): Orography of low-res data (same grid as low_res).

    Returns:
        corrected_temperature (xarray.DataArray or Dataset): Regridded and height-corrected temperature field.
    """

    high_res_orog = high_res_orog.sel(lat = slice(high_res.lat[0], high_res.lat[-1]))

    lapse_rate = 6.5  #        lapse_rate (float): Lapse rate in K/km (default: 6.5).

    interpolated_values = matching_resolutions_xe(low_res, high_res, method)
    interpolated_orog = matching_resolutions_xe(low_res_orog, high_res_orog, method)

    # Apply height correction (convert lapse rate from K/km to K/m)
    lapse_rate_per_m = lapse_rate / 1000.0
    delta_h = high_res_orog - interpolated_orog # low_res_orog - interpolated_orog

    # these should be exactly the same but sometimes a bit off, so forcing htem to be equal
    delta_h['lat'] = interpolated_values.lat.values
    delta_h['lon'] = interpolated_values.lon.values

    corrected_temp = interpolated_values + lapse_rate_per_m * delta_h





    # lapse_rate_per_m = lapse_rate / 1000.0
    # delta_h = high_res_orog - interpolated_orog
    # delta_h = delta_h.assign_coords(lat=interpolated_temp.lat.values, lon=interpolated_temp.lon.values)

    # corrected_temp = interpolated_temp + lapse_rate_per_m * delta_h
    # corrected_temp = corrected_temp.where(~np.isnan(high_res_ds["zeta"]))


    return corrected_temp






# def matching_resolutions_xe_temp_corr_combined(
#     low_res,
#     high_res,
#     low_res_orog,
#     high_res_orog,
#     method="conservative_normed",
#     lapse_rate=6.5
# ):
#     """
#     Regrid low-resolution temperature and orography to the high-resolution grid
#     using nearest neighbour, apply land-sea masks, and correct temperature for
#     elevation differences.

#     Parameters
#     ----------
#     low_res : xr.DataArray
#         Low-resolution temperature field with land-sea mask already applied.
#     high_res : xr.DataArray
#         High-resolution temperature field or observational reference grid with
#         land-sea mask already applied.
#     low_res_orog : xr.DataArray
#         Orography on the low-resolution grid.
#     high_res_orog : xr.DataArray
#         Orography on the high-resolution grid.
#     method : str, optional
#         Regridding method for xesmf. Default is "nearest_s2d".
#     lapse_rate : float, optional
#         Temperature lapse rate in K/km. Default is 6.5.

#     Returns
#     -------
#     xr.DataArray
#         Low-resolution temperature regridded to the high-resolution grid and
#         corrected for height differences.
#     """

#     high_res_orog = high_res_orog.sel(lat = slice(high_res.lat[0], high_res.lat[-1]))

#     low_res_ds = low_res.to_dataset(name="zeta")
#     low_res_ds["mask"] = xr.where(~np.isnan(low_res_ds["zeta"]), 1, 0)

#     high_res_ds = high_res.to_dataset(name="zeta")
#     high_res_ds["mask"] = xr.where(~np.isnan(high_res_ds["zeta"]), 1, 0)

#     low_res_orog_ds = low_res_orog.to_dataset(name="zeta")
#     low_res_orog_ds["mask"] = xr.where(~np.isnan(low_res_orog_ds["zeta"]), 1, 0)

#     high_res_orog_ds = high_res_orog.to_dataset(name="zeta")
#     high_res_orog_ds["mask"] = xr.where(~np.isnan(high_res_orog_ds["zeta"]), 1, 0)

#     temp_regridder = xe.Regridder(low_res_ds, high_res_ds, method, ignore_degenerate=True, unmapped_to_nan=True)
#     orog_regridder = xe.Regridder(low_res_orog_ds, high_res_orog_ds, method, ignore_degenerate=True, unmapped_to_nan=True)

#     interpolated_temp = temp_regridder(low_res_ds["zeta"])
#     interpolated_orog = orog_regridder(low_res_orog_ds["zeta"])

#     interpolated_temp = interpolated_temp.where(~np.isnan(high_res_ds["zeta"]))
#     interpolated_orog = interpolated_orog.where(~np.isnan(high_res_orog_ds["zeta"]))

#     lapse_rate_per_m = lapse_rate / 1000.0
#     delta_h = high_res_orog - interpolated_orog
#     delta_h = delta_h.assign_coords(lat=interpolated_temp.lat.values, lon=interpolated_temp.lon.values)

#     corrected_temp = interpolated_temp + lapse_rate_per_m * delta_h
#     corrected_temp = corrected_temp.where(~np.isnan(high_res_ds["zeta"]))

#     return corrected_temp




## TESTING OLD CODE ### 
# def matching_resolutions_xe(low_res, high_res, method = "conservative_normed"):

#     low_res['lat'] = np.round(low_res.lat, 2)
#     low_res['lon'] = np.round(low_res.lon, 2)

#     low_res_lats = low_res['lat'].values
#     low_res_lons = low_res['lon'].values

#     output_grid = xr.Dataset(
#     {
#         "lat": (["lat"], low_res_lats, {"units": "degrees_north"}),
#         "lon": (["lon"], low_res_lons, {"units": "degrees_east"}),
#     }
#     )    
    
#     regridder = xe.Regridder(high_res, output_grid, method)
#     interpolated_values = regridder(high_res, keep_attrs=True)
    
#     return interpolated_values


# now to upscale the era5land to the same resolution as cordex
def matching_times(data, start, stop, freq, stat):
    data = data.sel(time = slice(start, stop))
    
    if stat == "mean":
        data = data.resample(time = freq).mean()
    if stat == "max":
        data = data.resample(time = freq).max()

    return data


def island_bounds_box():

    "Lat / lon boxes of all the islands under examination"

    island_bounds = {
        "HIE": ([27.6, 28.1], [-18.2, -17.7]),
        "PAL": ([28.4, 29], [-18, -17.5]),
        "GOM": ([27.8, 28.3], [-17.5, -17]),
        "TEN": ([27.8, 28.8], [-17, -16]),
        "GRA": ([27.5, 28.5], [-16, -15.2]),
        "FUE": ([27.8, 28.8], [-15, -13.5]),
        "LAN": ([28.8, 29.4], [-14, -13.2]),
        "REU": ([-21.6, -20.6], [55, 56]),
        "BAR": ([12.8, 13.6], [-59.8, -59]),
        "GRE": ([11.8, 12.5], [-61.9, -61.4]),
        "LUC": ([13.6, 14.4], [-61.2, -60.6]),
        "TRI": ([9.9, 10.9], [-62, -60.7]),
        "TOB": ([11, 11.4], [-60.9, -60.4])
    }

    return island_bounds

## prepare extreme quantile data


## prepare extreme quantile data
# 
#format_data(reference, "tas", island)

def format_data(data, var, island= None, 
                time_mean = True, quant = None,
                resample = True, time_slice = True,
                gridded = True, time_start = True,
                time_end = True):

    try:
        if time_slice:
            if island == "REU" and var in ["tas", "tmean"]:
                if gridded == True:
                    # we only have tmax since 2000 for REUNION for gridded
                    data = data.sel(time=slice("2000", "2014"))
            elif time_start and time_end: # == True
                # make sure same time range
                data = data.sel(time=slice("1990", "2014"))
            else:
                data = data.sel(time=slice(time_start, time_end))

        # Resample to 3-hour intervals if necessary
        # if int((pd.to_datetime(data.time[2].values) - 
        # pd.to_datetime(data.time[1].values)).total_seconds() / 3600) < 3:
        #     data = data.resample(time="3h").mean()
    
        # RESAMPLE TO MATCH GRIDDED OBSERVATIONS
        if resample:
            if var in ["tas", "tmean"] and island in ["TEN", "GRA", "FUE", "LAN", "REU"]:
                data = data.resample(time='d').max()
            elif var in ["tas", "tmean", "tos"] and island in ["LUC", "BAR", "GRE", "TOB", "TRI"]:
                data = data.resample(time='d').mean()
            elif var in ["pr", "prlr"]:
                # Daily mean for PRECIP and seasonal selection
                data = data.resample(time='d').mean()
            else:
                print('UNKNOWN VARIABLE!')

        # time mean has to be true to enter this loop
        if time_mean and quant == None:
            to_plot = data.mean(dim = "time", skipna = True)
        elif time_mean and quant != None:
            to_plot = data.quantile(quant, dim = "time", skipna = True)

        # not performing any aggregation in time 
        else: 
            to_plot = data

     
    except KeyError: 
        # already been converted to quantile, mean  etc. 
        to_plot = data

    return to_plot







"""
Spatiotemporally-connected marine heatwave (MHW) detection.

Fixes over the original per-grid-cell approach:

1. Grid cells that exceed their local P90 threshold on the same day AND are
   spatially adjacent are merged into a single "patch" for that day.
2. Patches are linked across days whenever they overlap in space at all,
   which is what lets one physical event be tracked as it drifts, grows
   and shrinks. This is implemented as 3-D (time, lat, lon) connected-
   component labelling using a structuring element that allows full
   spatial adjacency within a day, plus same-pixel adjacency between
   consecutive days (which is exactly what "patch overlaps with the
   previous day's patch" means).
3. An event does NOT end just because it temporarily shrinks to a small
   connected remnant. It ends once its area collapses to
   <= area_decay_frac (default 0.10, i.e. 90% of cells have dropped
   below their threshold) of the largest area it reached at any point in
   its life so far. Everything after that point is cut loose from the
   event; if it regrows into its own connected patch it is free to become
   a brand-new event.

`event_id` in the returned dataset is therefore globally unique -- unlike
the old grid-cell-vectorized version, event 3 always means the same
physical, spatially-connected event everywhere it appears.
"""


def _spatial_structure(connectivity: int = 1) -> np.ndarray:
    """3x3 structuring element for spatial adjacency.

    connectivity=1 -> rook/4-connectivity  (shares an edge: N, S, E, W)
    connectivity=2 -> queen/8-connectivity (also shares a corner)
    """
    return ndimage.generate_binary_structure(2, connectivity)


def _raw_spatiotemporal_labels(exceed: np.ndarray, connectivity: int = 1):
    """
    3-D connected-component labelling of a boolean (time, lat, lon) array.

    Two exceedance cells belong to the same raw component if they are
    spatially adjacent on the same day, OR if they are the very same
    (lat, lon) cell on two consecutive days. Because within-day cells are
    already fully linked by spatial adjacency, a single shared pixel
    between day t and day t+1 is enough to weld an entire patch on day t
    to an entire (possibly different-shaped) patch on day t+1 -- i.e.
    "any overlap between consecutive days" connectivity, without needing
    an explicit patch-overlap bookkeeping step.
    """
    spatial_struct = _spatial_structure(connectivity)
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1] = spatial_struct   # full spatial connectivity within a day
    struct[0, 1, 1] = True       # link to the same cell on day t-1
    struct[2, 1, 1] = True       # link to the same cell on day t+1
    labels, n_labels = ndimage.label(exceed, structure=struct)
    return labels, n_labels


def _label_events_with_decay(
    exceed: np.ndarray,
    min_duration: int,
    area_decay_frac: float,
    connectivity: int,
    max_passes: int = 200,
):
    """
    Core algorithm. `exceed` is a boolean (time, lat, lon) array.

    Iteratively:
      1. Find raw spatiotemporal connected components in whatever
         exceedance is still "unclaimed".
      2. For each raw component, walk forward in time tracking area(t)
         (# active cells that day) and the running max area seen so far.
      3. The component is truncated the first time area(t) falls to
         <= area_decay_frac * running_max. Everything from t_start up to
         (but not including) that day becomes a candidate event; if it's
         >= min_duration days long it's finalised as a real event.
      4. Whatever comes after the truncation point (if anything) is left
         unclaimed and goes through the loop again, so a heatwave that
         nearly dies and then re-intensifies is split into two separate
         events rather than one event with a misleading multi-year span.

    Returns
    -------
    event_id : int array, same shape as `exceed`, 0 = no event
    event_area : int array, same shape as `exceed` -- number of active
        grid cells *that day*, for whichever event each cell belongs to
    event_records : list of dicts, one per finalised event, with the
        time/area bookkeeping needed to build the summary table
    """
    T = exceed.shape[0]
    remaining = exceed.copy()
    event_id = np.zeros(exceed.shape, dtype=np.int32)
    event_area = np.zeros(exceed.shape, dtype=np.int32)
    event_records: list[dict] = []
    next_event_id = 0

    for _ in range(max_passes):
        if not remaining.any():
            break

        labels, n_raw = _raw_spatiotemporal_labels(remaining, connectivity)
        if n_raw == 0:
            break

        any_claimed = False
        flat_labels = labels.reshape(T, -1)

        for raw_id in range(1, n_raw + 1):
            comp_mask = labels == raw_id
            area_per_t = (flat_labels == raw_id).sum(axis=1)
            active_days = np.flatnonzero(area_per_t > 0)
            # Contiguous by construction: a component can't skip a day
            # and still be time-connected under this structuring element.
            t_start, t_end = active_days[0], active_days[-1]
            areas = area_per_t[t_start : t_end + 1]

            running_max = 0
            term_idx = None  # index (relative to t_start) where decay trips
            for i, a in enumerate(areas):
                running_max = max(running_max, a)
                if i > 0 and a <= area_decay_frac * running_max:
                    term_idx = i
                    break

            head_end = t_start + (term_idx - 1 if term_idx is not None else len(areas) - 1)
            duration_days = head_end - t_start + 1
            head_areas = areas[: term_idx if term_idx is not None else len(areas)]

            # The "head" (t_start..head_end) is always claimed this pass,
            # whether or not it ends up qualifying as a real event -- this
            # is what guarantees the loop makes progress and terminates.
            head_mask = comp_mask.copy()
            head_mask[head_end + 1 :] = False
            remaining[comp_mask & (np.arange(T)[:, None, None] <= head_end)] = False
            any_claimed = True

            if duration_days >= min_duration:
                next_event_id += 1
                event_id[head_mask] = next_event_id
                for t in range(t_start, head_end + 1):
                    event_area[t][comp_mask[t]] = area_per_t[t]
                event_records.append(
                    dict(
                        event_id=next_event_id,
                        t_start=t_start,
                        t_end=head_end,
                        duration=duration_days,
                        max_area_cells=int(head_areas.max()),
                        mean_area_cells=float(head_areas.mean()),
                    )
                )
            # The tail (head_end+1 .. t_end), if any, stays True in
            # `remaining` and will be re-labelled (as its own, disconnected
            # component) on the next pass.

        if not any_claimed:
            break

    return event_id, event_area, event_records


def detect_mhw_events(
    sst: xr.DataArray,
    threshold: xr.DataArray,
    min_duration: int = 5,
    area_decay_frac: float = 0.10,
    connectivity: int = 1,
    max_passes: int = 200,
) -> xr.Dataset:
    """
    Identify spatially-connected marine heatwave events.

    A day-t patch is the set of grid cells with sst > threshold that are
    mutually adjacent (per `connectivity`). Patches are linked across days
    whenever they overlap in space at all. An event lasts as long as that
    chain of overlapping patches persists, and formally ends once its area
    has collapsed to <= `area_decay_frac` of the largest area it ever
    reached (default 10%, i.e. it ends once at least 90% of its peak
    footprint has dropped back below threshold) -- not merely once every
    last grid cell has dropped out.

    Parameters
    ----------
    sst, threshold : xr.DataArray
        Must share (or be broadcastable to) dims including 'time' and two
        spatial dims (e.g. 'lat', 'lon'). Loaded into memory internally
        (connected-component labelling is inherently a whole-domain
        operation and can't be chunked per pixel like the old
        apply_ufunc version) -- for very large grids, subset spatially or
        run per region/year first.
    min_duration : int
        Minimum number of days an event's "head" (see area_decay_frac)
        must last to be kept.
    area_decay_frac : float
        Event ends the first day its active area drops to <= this
        fraction of its running maximum area (default 0.10 -> 90% drop).
    connectivity : int
        1 = rook/4-connectivity (shares an edge), 2 = queen/8-connectivity
        (also shares a corner). Defines "adjacent".
    max_passes : int
        Safety cap on the peel-and-relabel loop.

    Returns
    -------
    xr.Dataset with dims (time, <spatial dims>):
        - event_id        : globally-unique event label, 0 = no event
        - sst_mhw          : sst on event days, NaN elsewhere
        - event_duration   : total duration (days) of the event that
                              cell/day belongs to (constant within an event)
        - event_area       : number of active grid cells that particular
                              day, for whichever event the cell belongs to
        - event_max_area   : peak area (# grid cells) reached over the
                              event's whole life (constant within an event)
    """
    if "time" not in sst.dims:
        raise ValueError("sst must have a 'time' dimension")

    sst_b, threshold_b = xr.broadcast(sst, threshold)
    spatial_dims = [d for d in sst_b.dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(
            f"Expected exactly 2 spatial dims besides 'time', got {spatial_dims}"
        )

    # Work in (time, dim0, dim1) order internally regardless of the
    # dataset's native dim order, then transpose back at the end.
    sst_t = sst_b.transpose("time", *spatial_dims)
    thr_t = threshold_b.transpose("time", *spatial_dims)
    exceed = (sst_t > thr_t).values

    event_id_np, event_area_np, records = _label_events_with_decay(
        exceed, min_duration, area_decay_frac, connectivity, max_passes
    )

    coords = sst_t.coords
    dims = sst_t.dims
    event_id = xr.DataArray(event_id_np, coords=coords, dims=dims, name="event_id")
    event_area = xr.DataArray(event_area_np, coords=coords, dims=dims, name="event_area")

    event_duration_np = np.zeros_like(event_id_np, dtype=np.float64)
    event_max_area_np = np.zeros_like(event_id_np, dtype=np.float64)
    for rec in records:
        m = event_id_np == rec["event_id"]
        event_duration_np[m] = rec["duration"]
        event_max_area_np[m] = rec["max_area_cells"]
    event_duration = xr.DataArray(
        np.where(event_id_np > 0, event_duration_np, np.nan),
        coords=coords, dims=dims, name="event_duration",
    )
    event_max_area = xr.DataArray(
        np.where(event_id_np > 0, event_max_area_np, np.nan),
        coords=coords, dims=dims, name="event_max_area",
    )

    sst_mhw = sst_t.where(event_id > 0).rename("sst_mhw")

    out = xr.Dataset(
        {
            "sst_mhw": sst_mhw,
            "event_duration": event_duration,
            "event_area": event_area,
            "event_max_area": event_max_area,
            "event_id": event_id,
        }
    )
    # restore the caller's original spatial dim order
    return out.transpose(*sst_b.dims)


def summarize_mhw_events(mhw_ds: xr.Dataset, cell_area_km2: xr.DataArray | None = None) -> pd.DataFrame:
    """
    Collapse the output of `detect_mhw_events` into one row per physical
    (spatially-connected) event: timing, intensity, and size.

    Parameters
    ----------
    mhw_ds : xr.Dataset
        Output of `detect_mhw_events`.
    cell_area_km2 : xr.DataArray, optional
        Per-grid-cell area (same spatial dims as mhw_ds), e.g. computed
        from a regular lat/lon grid with cos(latitude) weighting. If
        given, adds max/mean area columns in km^2 alongside grid-cell
        counts.

    Returns
    -------
    pd.DataFrame, one row per event, with columns:
        event_id, start_time, end_time, mid_time, duration_days,
        mean_sst, max_sst,
        max_area_cells, mean_area_cells, footprint_cells,
        lat_min, lat_max, lon_min, lon_max, centroid_lat, centroid_lon
        (+ max_area_km2, mean_area_km2, footprint_km2 if cell_area_km2 given)
    """
    spatial_dims = [d for d in mhw_ds["event_id"].dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected exactly 2 spatial dims besides 'time', got {spatial_dims}")
    lat_dim, lon_dim = spatial_dims

    df = mhw_ds.to_dataframe().reset_index()
    df = df[df["event_id"] > 0]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_id", "start_time", "end_time", "mid_time", "duration_days",
                "mean_sst", "max_sst", "max_area_cells", "mean_area_cells",
                "footprint_cells", "lat_min", "lat_max", "lon_min", "lon_max",
                "centroid_lat", "centroid_lon",
            ]
        )

    # --- timing + intensity, one row per grid-cell-day ---
    grid_level = (
        df.groupby("event_id")
        .agg(
            start_time=("time", "min"),
            end_time=("time", "max"),
            duration_days=("event_duration", "first"),
            mean_sst=("sst_mhw", "mean"),
            max_sst=("sst_mhw", "max"),
            max_area_cells=("event_max_area", "first"),
            lat_min=(lat_dim, "min"),
            lat_max=(lat_dim, "max"),
            lon_min=(lon_dim, "min"),
            lon_max=(lon_dim, "max"),
        )
        .reset_index()
    )
    grid_level["mid_time"] = (
        grid_level["start_time"] + (grid_level["end_time"] - grid_level["start_time"]) / 2
    )

    # --- daily area, de-duplicated to one row per (event, day) so the
    # mean isn't biased towards days with more active grid cells ---
    day_level = df.drop_duplicates(subset=["event_id", "time"])[["event_id", "event_area"]]
    mean_area = day_level.groupby("event_id")["event_area"].mean().rename("mean_area_cells")

    # --- footprint: unique grid cells ever touched, and their centroid ---
    footprint = df.drop_duplicates(subset=["event_id", lat_dim, lon_dim])
    footprint_stats = footprint.groupby("event_id").agg(
        footprint_cells=(lat_dim, "size"),
        centroid_lat=(lat_dim, "mean"),
        centroid_lon=(lon_dim, "mean"),
    )

    summary = (
        grid_level.set_index("event_id")
        .join(mean_area)
        .join(footprint_stats)
        .reset_index()
    )

    if cell_area_km2 is not None:
        area_da = cell_area_km2.transpose(lat_dim, lon_dim)
        cell_km2 = dict(
            zip(
                zip(area_da[lat_dim].values.repeat(area_da.sizes[lon_dim]),
                    np.tile(area_da[lon_dim].values, area_da.sizes[lat_dim])),
                area_da.values.ravel(),
            )
        )
        avg_cell_km2 = float(np.nanmean(area_da.values))
        summary["max_area_km2"] = summary["max_area_cells"] * avg_cell_km2
        summary["mean_area_km2"] = summary["mean_area_cells"] * avg_cell_km2
        summary["footprint_km2"] = summary["footprint_cells"] * avg_cell_km2

    cols = [
        "event_id", "start_time", "end_time", "mid_time", "duration_days",
        "mean_sst", "max_sst",
        "max_area_cells", "mean_area_cells", "footprint_cells",
        "lat_min", "lat_max", "lon_min", "lon_max", "centroid_lat", "centroid_lon",
    ]
    if cell_area_km2 is not None:
        cols += ["max_area_km2", "mean_area_km2", "footprint_km2"]
    return summary[cols].sort_values("start_time").reset_index(drop=True)






















    
# def matching_resolutions(low_res, high_res):
    
#     # Assuming low_res_data and high_res_data are xarray.DataArray objects
#     # Extract coordinates
#     try:
#         low_res_lats = low_res['rlat'].values
#         low_res_lons = low_res['rlon'].values
#     except KeyError:
#         low_res_lats = low_res['lat'].values
#         low_res_lons = low_res['lon'].values
        
#     high_res_lats = high_res['lat'].values
#     high_res_lons = high_res['lon'].values
#     try:
#         time_values = high_res['time'].values
#     except KeyError:
#         time_values = 0

#     # Flatten high-res lat/lon grid and data
#     high_res_points = np.array([(lat, lon) for lat in high_res_lats for lon in high_res_lons])
#     high_res_values = high_res.values.flatten()

#     # Create low-res lat/lon meshgrid
#     low_res_lon_grid, low_res_lat_grid = np.meshgrid(low_res_lons, low_res_lats)
#     low_res_points = np.array([low_res_lat_grid.ravel(), low_res_lon_grid.ravel()]).T

#     # Placeholder for interpolated timesteps
#     interpolated_timesteps = []

#     runto = len(high_res)



#     if time_values == 0:
#         high_res_values = high_res.values.flatten()

#         interpolated_values = griddata(
#             points=high_res_points,
#             values=high_res_values,
#             xi=low_res_points,
#             method='nearest'
#         )

#         interpolated_values = interpolated_values.reshape(low_res_lat_grid.shape)
#         # Convert to xarray DataArray
#         interpolated_data_array = interpolated_values

#     else:
#         for t in range(runto):
#             high_res_slice = high_res.isel(time=t)
#             high_res_values = high_res_slice.values.flatten()

#             interpolated_values = griddata(
#                 points=high_res_points,
#                 values=high_res_values,
#                 xi=low_res_points,
#                 method='nearest'
#             )

#             interpolated_values = interpolated_values.reshape(low_res_lat_grid.shape)

#             interpolated_timesteps.append(interpolated_values)

#         # Convert to xarray DataArray
#         interpolated_data_array = np.stack(interpolated_timesteps, axis=0)

#     if time_values == 0:

#         interpolated_data = xr.DataArray(
#             interpolated_data_array,
#             dims=["lat", "lon"],
#             coords={
#                 "lat": low_res_lats,
#                 "lon": low_res_lons
#             },
#             name="interpolated"
#         )

#     else:
#         interpolated_data = xr.DataArray(
#             interpolated_data_array,
#             dims=["time", "lat", "lon"],
#             coords={
#                 "time": time_values[0:runto],
#                 "lat": low_res_lats,
#                 "lon": low_res_lons
#             },
#             name="interpolated"
#         )
#     return interpolated_data


# based on this cycle, mark the heatwaves by also lasting for 5 consecative days of more. 

# def detect_mhw_events_1D(sst, threshold, min_duration=5):
#     """
#     Identify marine heatwave events as runs of >= min_duration consecutive
#     days where sst > threshold.

#     Parameters
#     ----------
#     sst : xr.DataArray
#         Daily SST time series (1D, dim='time').
#     threshold : xr.DataArray
#         Same-length daily threshold time series (e.g. nemo_tos_threshold_full).
#     min_duration : int
#         Minimum number of consecutive days to qualify as an MHW event.

#     Returns
#     -------
#     xr.Dataset with:
#         - sst_mhw: SST values on MHW days, NaN elsewhere
#         - event_duration: total duration (days) of the event that day belongs to, NaN elsewhere
#         - event_id: integer event label (1, 2, 3, ...), 0 where no event
#     """
#     try: 
#         sst = sst.mean(dim = ["lat", "lon"])
#     except ValueError:
#         sst = sst 

    
#     exceed = (sst > threshold).values  # boolean array, True where sst exceeds threshold
#     n = len(exceed)

#     event_id = np.zeros(n, dtype=int)
#     duration = np.full(n, np.nan)

#     current_event = 0
#     i = 0
#     while i < n:
#         if exceed[i]:
#             start = i
#             while i < n and exceed[i]:
#                 i += 1
#             length = i - start
#             if length >= min_duration:
#                 current_event += 1
#                 event_id[start:i] = current_event
#                 duration[start:i] = length
#         else:
#             i += 1

#     # SST values only on MHW days, NaN elsewhere
#     sst_mhw = sst.where(event_id > 0)

#     # Duration DataArray: same value repeated for every day within an event
#     event_duration = xr.DataArray(
#         duration,
#         coords={"time": sst.time},
#         dims="time",
#         name="event_duration"
#     )

#     event_id_da = xr.DataArray(
#         event_id,
#         coords={"time": sst.time},
#         dims="time",
#         name="event_id"
#     )

#     ds_out = xr.Dataset({
#         "sst_mhw": sst_mhw,
#         "event_duration": event_duration,
#         "event_id": event_id_da
#     })

#     return ds_out

# # THIS IS WRONG! REMOVES THE SPATIAL DIMENSION
# def summarize_mhw_events(mhw_ds):
#     """
#     Collapse the daily MHW dataset (from detect_mhw_events) into one row per event:
#     midpoint time, mean SST during the event, and duration.
#     """
#     df = mhw_ds.to_dataframe().reset_index()
#     df = df[df["event_id"] > 0]  # drop non-event days

#     summary = (
#         df.groupby("event_id")
#         .agg(
#             start_time=("time", "min"),
#             end_time=("time", "max"),
#             mean_sst=("sst_mhw", "mean"),
#             duration=("event_duration", "first"),  # constant within an event
#         )
#         .reset_index()
#     )

#     # Midpoint time for plotting each event as a single bubble
#     summary["mid_time"] = summary["start_time"] + (summary["end_time"] - summary["start_time"]) / 2

#     return summary



# def _detect_1d_run(exceed_1d, min_duration):
#     """
#     Vectorized run-length detector for a single 1D boolean time series.
#     Returns (event_id, duration) arrays of the same length.
#     """
#     n = exceed_1d.shape[0]
#     event_id = np.zeros(n, dtype=int)
#     duration = np.full(n, np.nan)

#     if not np.any(exceed_1d):
#         return event_id, duration

#     # pad with False on both ends so runs touching the edges are detected
#     padded = np.concatenate(([False], exceed_1d, [False]))
#     diff = np.diff(padded.astype(np.int8))
#     starts = np.flatnonzero(diff == 1)       # inclusive run start indices
#     ends = np.flatnonzero(diff == -1)        # exclusive run end indices

#     current_event = 0
#     for s, e in zip(starts, ends):
#         length = e - s
#         if length >= min_duration:
#             current_event += 1
#             event_id[s:e] = current_event
#             duration[s:e] = length

#     return event_id, duration


# def detect_mhw_events(sst, threshold, min_duration=5):
#     """
#     Identify marine heatwave events as runs of >= min_duration consecutive
#     days where sst > threshold, at every grid point independently.

#     Parameters
#     ----------
#     sst : xr.DataArray
#         Daily SST with a 'time' dimension. May also have 'lat'/'lon' (or
#         any other spatial dims) — a plain 1D time series still works.
#     threshold : xr.DataArray
#         Daily threshold, broadcastable against `sst` (e.g. a 'time'-only
#         climatological threshold, or one with matching spatial dims).
#     min_duration : int
#         Minimum number of consecutive days to qualify as an MHW event.

#     Returns
#     -------
#     xr.Dataset with, per grid cell:
#         - sst_mhw: SST values on MHW days, NaN elsewhere
#         - event_duration: duration (days) of the event that day belongs
#           to, NaN elsewhere
#         - event_id: integer event label (1, 2, 3, ...) per pixel, 0 where
#           no event
#     """
#     if "time" not in sst.dims:
#         raise ValueError("sst must have a 'time' dimension")

#     sst_b, threshold_b = xr.broadcast(sst, threshold)
#     exceed = sst_b > threshold_b

#     event_id, duration = xr.apply_ufunc(
#         _detect_1d_run,
#         exceed,
#         kwargs={"min_duration": min_duration},
#         input_core_dims=[["time"]],
#         output_core_dims=[["time"], ["time"]],
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[int, float],
#     )

#     # apply_ufunc moves 'time' to the end; put dims back in the original order
#     event_id = event_id.transpose(*sst_b.dims).rename("event_id")
#     duration = duration.transpose(*sst_b.dims).rename("event_duration")
#     sst_mhw = sst_b.where(event_id > 0).rename("sst_mhw")

#     return xr.Dataset({
#         "sst_mhw": sst_mhw,
#         "event_duration": duration,
#         "event_id": event_id,
#     })


# def get_season(month):
#     if month in [12, 1, 2]:
#         return "DJF"
#     elif month in [3, 4, 5]:
#         return "MAM"
#     elif month in [6, 7, 8]:
#         return "JJA"
#     else:
#         return "SON"

# def summarize_period(event_summary, start_year, end_year):
#     """
#     Compute MHW summary statistics for events falling within [start_year, end_year] inclusive,
#     based on event start_time.
#     """
#     period_events = event_summary[
#         (event_summary["start_time"].dt.year >= start_year) &
#         (event_summary["start_time"].dt.year <= end_year)
#     ].copy()

#     n_events = len(period_events)
#     total_mhw_days = period_events["duration"].sum()

#     # Total days in period (accounts for leap years)
#     period_start = pd.Timestamp(f"{start_year}-01-01")
#     period_end = pd.Timestamp(f"{end_year}-12-31")
#     total_period_days = (period_end - period_start).days + 1

#     pct_days_mhw = 100 * total_mhw_days / total_period_days

#     # Seasonal mean temperature of MHW events (based on event start month)
#     period_events["season"] = period_events["mid_time"].dt.month.apply(get_season)
#     seasonal_mean_temp = period_events.groupby("season")["mean_sst"].mean()
#     seasonal_mean_temp = seasonal_mean_temp.reindex(["DJF", "MAM", "JJA", "SON"])

#     seasonal_event_count = period_events.groupby("season").size()
#     seasonal_event_count = seasonal_event_count.reindex(["DJF", "MAM", "JJA", "SON"], fill_value=0)

#     overall_mean_temp = period_events["mean_sst"].mean()
#     mean_duration = period_events["duration"].mean()

#     return {
#         "n_events": n_events,
#         "total_mhw_days": total_mhw_days,
#         "total_period_days": total_period_days,
#         "pct_days_mhw": pct_days_mhw,
#         "overall_mean_temp": overall_mean_temp,
#         "mean_duration": mean_duration,
#         "seasonal_mean_temp": seasonal_mean_temp,
#         "seasonal_event_count": seasonal_event_count,
#     }

