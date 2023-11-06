"""
See documentation at: 
https://github.com/swamidasslab/tiffslide-xarray
"""


from .data import load_tiff_level, open_all_levels, TiffBackendEntrypoint
from ._version import __version__

# Easier aliases
open_level = load_tiff_level
open = open_all_levels
