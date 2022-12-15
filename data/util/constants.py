"""Define constants to be used throughout the repository."""

# Main paths

# Dataset constants 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# US latitude/longitude boundaries
US_N = 49.4
US_S = 24.5
US_E = -66.93
US_W = -124.784

# Test image
TEST_IMG_PATH = [".circleci/images/test_image.png"] * 2

# List of class labels
class_labels_list = ["CAFOs", "Mines", "WWTreatment", "RefineriesAndTerminals", "Landfills", "ProcessingPlants"]

# These constants are used for writing results to CSV
metrics = ['accuracy', 'auprc', 'auroc', 'f1', 'precision', 'prevalence', 'recall', 'threshold']

# Image Metadata
COL_IMG_LAT = 'lat'
COL_IMG_LON = 'lon'
COL_IMG_NEG_SRC = 'negative_source'

# Finding negatives
SIM_SEARCH_PREFIX = 'https://search.descarteslabs.com/search?'
N_SIM_DIFFICULT = 1000
SIM_MODEL_TILE_SZ = 64
SIM_MODEL_RES = 1
SIM_MODEL_PAD = 32

# Size of images to feed into model
TILESIZE = 720

# List of product/band combinations to select from
valid_products = ['naip', 'naip-rgb', 'sentinel2', 'sentinel2-rgb', 'sentinel1', 'all', 'sentinels'] 

# Data ranges
NAIP_MIN = 0
NAIP_MAX = 255
S2_RGB_MIN = 0
S2_RGB_MAX = 4000
S2_NIR_MIN = 0
S2_NIR_MAX = 10000
S1_VH_MIN = 585
S1_VH_MAX = 2100
S1_VV_MIN = 585
S1_VV_MAX = 2926
