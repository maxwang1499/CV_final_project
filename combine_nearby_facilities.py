"""
Command-line tool to operate on csv files

:usage:
    $ python -m meter_util.combine_nearby_facilities -i <infile> -o <outfile>
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import networkx as nx
from shapely.geometry import Point

# The max distance in km between points
# Images are 750 meters by 750 meters
HAVERSINE_THRESHOLD = 0.375

def haversine(lat1, lon1, lat2, lon2):
    """Obtain haversize distance for arrays of lat and lon.
    Args:
            lat1 (arr): latitudes of first array
            lon1 (arr): longitudes of first array
            lat2 (arr): latitudes of second array
            lon2 (arr): longitudes of second array 
    Returns:
            An array of haversine distances in km between each pair of lat&lon in the input arrays
    """
    lat1 = lat1*np.pi/180.0
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    d = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin((lon2 - lon1)/2)**2

    return 2 * 6371 * np.arcsin(np.sqrt(d))

def combine_nearby_facilities(df):
    """Combine dataframe of sources where lat&lon points that are near each other.
    Args:
            df: dataframe to combine sources
    Returns:
            A new combined dataframe
    """
    lats = df['Latitude'].values
    lons = df['Longitude'].values
    to_drop = []
    counter = 1
    for i, row in df.iterrows():
        if i % 10000 == 0:
            print("Iterating at row ", i)
            
        lat, lon = row['Latitude'], row['Longitude']
        
        # Computing haversine distance over all boxes is slow
        # Only consider boxes within 0.1 lat/lon
        lat_is_close = np.isclose(lats, lat, rtol=0, atol=0.1)
        lon_is_close = np.isclose(lons, lon, rtol=0, atol=0.1)
        lat_and_lon_are_close = lat_is_close & lon_is_close
        distances = haversine(
            lats[lat_and_lon_are_close],
            lons[lat_and_lon_are_close],
            [lat], [lon]
        )
        
        # Only take points within HAVERSINE_THRESHOLD 
        lat_lon_is_close = np.where(distances < HAVERSINE_THRESHOLD)[0]
        neighbors = np.where(lat_and_lon_are_close)[0][lat_lon_is_close]

        rows = [df.loc[neighbor] for neighbor in neighbors]
        if len(rows) > 1:
            df_to_modify = pd.DataFrame(rows)
            
            # Take all the unique nearby sources
            closeby_sources = set(df_to_modify.Source.str.cat(sep='-').split('-'))
            
            # Take all the unique previous sources
            prev_sources = set(df.loc[i, ["Source"]].str.cat(sep='-').split('-'))
            df.loc[i, ["Source"]] = '-'.join(closeby_sources | prev_sources)
            
            closeby_types = set(df_to_modify.Type.str.cat(sep='-').split('-'))
            prev_types = set(df.loc[i, ["Type"]].str.cat(sep='-').split('-'))
            df.loc[i, ["Type"]] = '-'.join(closeby_types | prev_types)

            counter += 1

    
    print(counter, "images had multiple facility types out of", df.shape[0], "images.")

    return df


def main():
    prog = 'python -m meter_util.combine_nearby_facilities -i <infile> -o <outfile>'
    description = ('A simple command line interface for csv files '
                   'to combine longtitude and latitude points and souces within a range.')
    
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument('-i', '--infile', nargs='?',
                        help='input file',
                        default=None)
    parser.add_argument('-o', '--outfile', nargs='?',
                        help='output file',
                        default=None)
    
    options = parser.parse_args()
    if not options.infile or not options.outfile:
        parser.print_help()
        sys.exit()

    try:
        df = pd.read_csv(options.infile)
        
        # We currently drop these columns as they're unnecessary
        new_df = df.drop(columns=['Unnamed: 0', 'geometry'])#, 'Date', 'Address'])
        
        combined_df = combine_nearby_facilities(new_df)


        # Output csv
        combined_df.to_csv(options.outfile)
    except ValueError as e:
        raise SystemExit(e)


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError as exc:
        sys.exit(exc.errno)
