"""
See documentation at: 
https://github.com/swamidasslab/tiffslide-xarray
"""

import tiffslide
import numpy as np
import xarray as xr
from datatree import DataTree
from typing import Optional, Any
import os

from xarray.backends import BackendArray, CachingFileManager, BackendEntrypoint


from xarray.core import indexing

def load_tiff_level(
      fname: str,
      level : int = 0, 
      tifffile_options : Optional[dict[str, Any]] = None,
      storage_options : Optional[dict[str, Any]] = None
  ) -> xr.Dataset:
  """Load specific level of a tiff slide into a xarray.Datset."""

  file_manager = CachingFileManager(
    tiffslide.TiffSlide,
    fname,
    kwargs=dict(tifffile_options=tifffile_options, storage_options=storage_options)
  )

  return _load_tiff_level(file_manager, fname, level)

def open_all_levels(
      fname: str,
      tifffile_options : Optional[dict[str, Any]] = None,
      storage_options : Optional[dict[str, Any]] = None
    ) -> DataTree:
  """Load all levels of a tiff slide into a datatree.DataTree."""

  file_manager = CachingFileManager(
    tiffslide.TiffSlide,
    fname,
    kwargs=dict(tifffile_options=tifffile_options, storage_options=storage_options)
  )

  tree = {}
  with file_manager.acquire_context() as f:
    n_levels = len(f.level_downsamples) #type: ignore
  
  for level in range(n_levels):

    x = _load_tiff_level(file_manager, fname, level)   

    tree["/" if level == 0 else f"level{level}"] = x

  tree = DataTree.from_dict(tree)
  return tree


def _load_tiff_level(
      file_manager: CachingFileManager, fname, level : int = 0, name="image") -> xr.Dataset:
  """Eagerly load a particular level of a tiff slide. Add coordinates, attributes, and set encodings
  to reasonable defaults."""
  with file_manager.acquire_context() as f:
    n_levels = len(f.level_dimensions)  #type: ignore
    if level < 0: level = n_levels + level

    shape =  f.level_dimensions[level] #type: ignore
    downsample = f.level_downsamples[level] #type: ignore
    attr = {k: v for k,v in f.properties.items() if (v != None and "tiffslide.level" not in k)} #type: ignore

  x = TiffSlideArray(file_manager, level)
  x = indexing.LazilyIndexedArray(x)

  if downsample== int(downsample):
    downsample = int(downsample)

  offset = (downsample - 1) / 2 if downsample > 1 else 0
  stride = downsample

  coords = {
      'y': ('y', np.arange(shape[1]) * stride + offset),
      'x': ('x', np.arange(shape[0]) * stride + offset),
      'c': ("c", ['r', 'g', 'b']),
    }

  x = xr.DataArray(
    x,
    dims=('y', 'x', 'c'), 
    coords=coords)

  x.attrs["level"] = level
  x.encoding["chunksizes"] = (256, 256, 1)
  x.encoding["compression"] = "lzf"

  

  x = x.to_dataset(name=name)
  x.attrs.update(attr)

  if type(fname) == str:
    x.attrs["source_file"] = fname
    x[name].attrs["source_file"] = fname

  for c in 'xy':
    x[c].encoding["scale_factor"] = stride
    x[c].encoding["add_offset"] = offset
    x[c].encoding["compression"] = "lzf"
    x[c].attrs["units"] = "pixel"
    try:
      x[c].attrs["mpp"] = x.attrs[f'tiffslide.mpp-{c}']
    except KeyError: pass
  
  

  return x



class TiffBackendEntrypoint(BackendEntrypoint):
    """Add entry point for xarray so that xarray.open_dataset
    can lazy load tiff and svs files using tiffslide.
    
    level: default 0, specifies which level of the image is read.

    tifffile_options: default None, is dict of keyword arguments to tifffile.

    storage_options: Default, None, is dict of keyword arguments to fsspec.
    """
  
    open_dataset_parameters : list  = ["level", "tifffile_options", "storage_options"]
    description : str  = "Load any image file compatible with Tiffslide."
    EXTENSIONS : set[str] = {".svs", ".tiff"}

    def open_dataset(
        self,
        filename_or_obj,
        level : Optional[int] = 0,
        tifffile_options : Optional[dict[str, Any]] = None,
        storage_options : Optional[dict[str, Any]] = None,
        *,
        drop_variables = None,
    ):
        assert type(level) == int
        return load_tiff_level(
          filename_or_obj, 
          level, 
          tifffile_options=tifffile_options,
          storage_options=storage_options)
        

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj) #type: ignore
        except TypeError:
            return False
        return ext in self.EXTENSIONS


class TiffSlideArray(BackendArray):
    def __init__(
        self,
        file : CachingFileManager, 
        level = 0,
    ):
        self.dtype = np.uint8
        self.level = level
        self.file = file        

        with self.file.acquire_context() as slide:
          dims = slide.level_dimensions[level] #  type: ignore (x, y)
          self.shape = (dims[1], dims[0], 3) # (y, x, c)
      

    def __getitem__(
        self, key: indexing.ExplicitIndexer
    ): # -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) :#-> np.typing.ArrayLike:
        # thread safe method that access to data on disk

        minmax = []
        offset_key = []

        for n, k in enumerate(key[:2]):
          if type(k) == int:
            if k < 0: k = self.shape[n] - k
            minmax.append((k, k))
            offset_key.append(0)

          if type(k) == slice:
            start = k.start if k.start is not None else 0
            stop = k.stop if k.stop is not None else self.shape[n] - 1

            minmax.append((start, stop))


            offset_key.append(
              slice(
                k.start - start if start else start, 
                k.stop - start if start else k.stop,
                k.step)
              )

        offset_key = tuple(offset_key + list(key[2:]))
            
        with self.file.acquire_context() as slide:
          region = slide.read_region( #type: ignore
            (
              minmax[1][0], # origin (x, y)
              minmax[0][0]
            ), 
            self.level, 
            (
              minmax[1][1] - minmax[1][0] + 1,  # size (x, y)
              minmax[0][1] - minmax[0][0] + 1
            ), 
            as_array=True)
        
        out =  region[offset_key]

        return out
            
        
              
