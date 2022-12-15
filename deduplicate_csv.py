"""
Command-line tool to operate on csv files

:usage:
    $ python -m meter_util.deduplicate_csv -i <infile> -o <outfile>
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
HAVERSINE_THRESHOLD = 0.01

def haversine(lat1, lon1, lat2, lon2):
    """obtain haversize distance for arrays of lat and lon.
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

def deduplicate_df(df):
    """deduplicate dataframe of lat&lon points that are near each other.
    Args:
            df: dataframe to deduplicate
    Returns:
            A new dedeuplicated dataframe
    """
    lats = df['Latitude'].values
    lons = df['Longitude'].values
    to_drop = []
    for i, row in df.iterrows():
        if i not in to_drop:
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
            
            # Drop every neighbor other than the first row being modified 
            to_drop += list(np.setdiff1d(neighbors, np.array([i])).flatten())

            rows = [df.loc[neighbor] for neighbor in neighbors]
            if len(rows) > 1:
                df_to_modify = pd.DataFrame(rows)
                df.loc[i, ["Latitude"]] = df_to_modify["Latitude"].mean()
                df.loc[i, ["Longitude"]] = df_to_modify["Longitude"].mean()
                df.loc[i, ["Source"]] = '-'.join(df_to_modify['Source'].unique())
                #TODO: Modify Date Column ?
    
    print("Removed ", len(set(to_drop)), " from ", df.shape[0], " datapoints ")

    return df.drop(index=set(to_drop))


def main():
    prog = 'python -m meter_util.deduplicate_csv -i <infile> -o <outfile>'
    description = ('A simple command line interface for csv files '
                   'to deduplicate longtitude and latitude points within a range.')
    
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
        processed_df_list = []
        df = pd.read_csv(options.infile)
        types = df['Type'].unique()
        for source_type in types:
            print("Deduplicating ", source_type)

            # Deduplicate df by specific source
            temp_df = df[df.Type == source_type].reset_index()
            deduplicated_df = deduplicate_df(temp_df)
            processed_df_list.append(deduplicated_df)
        
        # Concat everything together and drop unneccesary columns
        main_df = pd.concat(processed_df_list).drop(columns=['index', 'Unnamed: 0'])

        # Output csv
        main_df.to_csv(options.outfile)
    except ValueError as e:
        raise SystemExit(e)


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError as exc:
        sys.exit(exc.errno)
