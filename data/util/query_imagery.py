import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('')).parent.parent))

import json
import requests
import subprocess
import numpy as np

import descarteslabs as dl

from util import constants as C

def sim_search_request(query_tile, n):
    """Query for n images similar to {query_tile} using DL similarity search."""
    
    sim_search_suffix = f'key={query_tile.key}&layer=naip_v2_rgb_2017-2022&limit={n}'
    sim_search_url = C.SIM_SEARCH_PREFIX + sim_search_suffix
    r = requests.get(sim_search_url)
    
    if r.status_code == 401:
        print(f"Navigate to {sim_search_url}")
        print("and paste data here:")
        
        # Change terminal behavior to allow for > 4095 chars entered
        subprocess.check_call(["stty","-icanon"])
        json_obj = json.loads(input())
        subprocess.check_call(["stty","icanon"])
    else:
        json_obj = r.json()
    return json_obj

    
def search_similar_images(lat, lon, n):
    """Search for images with similar characteristics to (lat, lon)
    
    Args:
        lat {float} -- latitude of target landscape
        lon {float} -- longitude of target landscape
        n {int} -- number of similar images to flex
        
    Returns:
        {list of dictionaries} -- coordinates of similar images
    
    """
    print(f'- Fetching {n} images similar to coordinate ({lat}, {lon})')
    
    # Use fixed parameters in order to match similarity model
    query_tile = dl.scenes.DLTile.from_latlon(
        lat=lat,
        lon=lon,
        tilesize=C.SIM_MODEL_TILE_SZ,
        resolution=C.SIM_MODEL_RES,
        pad=C.SIM_MODEL_PAD
    )
    json_obj = sim_search_request(query_tile, n)

    similar_images = []
    for image in json_obj['features']:
        coords = image['geometry']['coordinates'][0]
        lon = (coords[1][0] + coords[3][0]) / 2.0
        lat = (coords[0][1] + coords[2][1]) / 2.0
        
        similar_images.append({
            'lat': lat,
            'lon': lon
        })
    return similar_images