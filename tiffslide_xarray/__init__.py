"""
See documentation at: 
https://github.com/swamidasslab/tiffslide-xarray
"""


from . import accessors
from .data import load_tiff_level, open_all_levels, TiffBackendEntrypoint


# Easier aliases
open_level = load_tiff_level
open = open_all_levels