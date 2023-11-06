from __future__ import annotations


import numpy as np
import xarray as xr
from datatree import DataTree
from typing import Optional, Any, NamedTuple, Callable
import os

from xarray.backends import BackendArray, CachingFileManager, BackendEntrypoint
from xarray.core import indexing



def load_tiff_level(
    fname: str,
    level: int = 0,
    tifffile_options: Optional[dict[str, Any]] = None,
    storage_options: Optional[dict[str, Any]] = None,
) -> xr.Dataset:
    """Load specific level of a tiff slide into a xarray.Datset."""

    file_manager = CachingFileManager(
        _zarr_tiffslide_opener,
        fname,
        kwargs=dict(tifffile_options=tifffile_options, storage_options=storage_options),
    )

    return _load_tiff_level(file_manager, fname, level)


def open_all_levels(
    fname: str,
    tifffile_options: Optional[dict[str, Any]] = None,
    storage_options: Optional[dict[str, Any]] = None,
    level_to_group: Callable[[int], str] = lambda level: "/" if not level else f"level{level}",
) -> DataTree:
    """Load all levels of a tiff slide into a datatree.DataTree."""

    file_manager = CachingFileManager(
        _zarr_tiffslide_opener,
        fname,
        kwargs=dict(tifffile_options=tifffile_options, storage_options=storage_options),
    )

    tree = {}
    with file_manager.acquire_context() as (zarr, f):
        n_levels = len(f.level_downsamples)  # type: ignore

    for level in range(n_levels):
        x = _load_tiff_level(file_manager, fname, level)

        #tree["/" if level == 0 else f"level{level}"] = x
        tree[level_to_group(level)] = x

    tree = DataTree.from_dict(tree)
    return tree


def _load_tiff_level(
    file_manager: CachingFileManager, fname, level: int = 0, name="image"
) -> xr.Dataset:
    """Lazy load a particular level of a tiff slide. Add coordinates, attributes, and set encodings
    to reasonable defaults."""
    import tiffslide
    
    with file_manager.acquire_context() as (zarr, slide):  # type: ignore
        f: tiffslide.TiffSlide = slide  # type: ignore
        n_levels = len(f.level_dimensions)  # type: ignore

        if level < 0:
            level = n_levels + level

        zarr: xr.DataArray = zarr[str(level)]

        shape = zarr.shape
        dims = [d[0].lower() for d in zarr.dims]  # type: ignore
        downsample = f.level_downsamples[level]
        encoding = zarr.encoding
        dataset_attr = {
            k: v
            for k, v in f.properties.items()
            if (v != None and "tiffslide.level" not in k)
        }

        array_attrs = zarr.attrs

    if downsample == int(downsample):
        downsample = int(downsample)

    offset = (downsample - 1) / 2 if downsample > 1 else 0
    stride = downsample

    coords = {}

    for d, s in zip(dims, shape):
        if d in "xyz":  # type:ignore
            coords[d] = np.arange(s) * stride + offset

    x = TiffSlideArray(file_manager, level)
    x = indexing.LazilyIndexedArray(x)
    x = xr.DataArray(
        x,
        dims=dims,
        coords=coords,
        attrs=array_attrs,
    )

    x.encoding = encoding
    x.encoding["_Unsigned"] = True
    x.encoding["preferred_chunks"] = {
        k[0].lower(): v for k, v in x.encoding["preferred_chunks"].items()
    }
    

    for d in x.coords:
        if d in set("xyz"):
            x.coords[d].encoding["scale_factor"] = stride
            x.coords[d].encoding["add_offset"] = offset
            x.coords[d].encoding["compression"] = "lzf"

            x.coords[d].attrs["units"] = "px"
            try:
                x.coords[d].attrs["mpp"] = dataset_attr[f"tiffslide.mpp-{d.lower()}"]
            except KeyError:
                pass

    x.attrs["level"] = level

    x = x.to_dataset(name=name)
    x.attrs.update(dataset_attr)

    if type(fname) == str:
        x.attrs["source_file"] = fname
        x[name].attrs["source_file"] = fname

    try:
        x.attrs["mpp"] = x.attrs[f"tiffslide.mpp"]
    except KeyError:
        pass

    return x


class TiffBackendEntrypoint(BackendEntrypoint):
    """Add entry point for xarray so that xarray.open_dataset
    can lazy load tiff and svs files using tiffslide.

    level: default 0, specifies which level of the image is read.

    tifffile_options: default None, is dict of keyword arguments to tifffile.

    storage_options: Default, None, is dict of keyword arguments to fsspec.
    """

    open_dataset_parameters: list = ["level", "tifffile_options", "storage_options", ""]
    description: str = "Load any image file compatible with Tiffslide."
    EXTENSIONS: set[str] = {".svs", ".tiff", ".tif"}

    def open_dataset(
        self,
        filename_or_obj,
        level: Optional[int] = 0,
        tifffile_options: Optional[dict[str, Any]] = None,
        storage_options: Optional[dict[str, Any]] = None,
        *,
        drop_variables=None,
    ):
        assert type(level) == int
        return load_tiff_level(
            filename_or_obj,
            level,
            tifffile_options=tifffile_options,
            storage_options=storage_options,
        )

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)  # type: ignore
        except TypeError:
            return False
        return ext.lower() in self.EXTENSIONS


class TiffSlideArray(BackendArray):
    def __init__(
        self,
        file: CachingFileManager,
        level=0,
    ):
        self.level = level
        self.file = file

        with self.file.acquire_context() as (zarr, slide):
            x: xr.DataArray = zarr[str(self.level)]  # type: ignore
            self.shape = x.shape
            self.dtype = x.dtype

    def get_array(self) -> indexing.ExplicitlyIndexedNDArrayMixin:
        with self.file.acquire_context() as (zarr, slide):
            x: xr.DataArray = zarr[str(self.level)]  # type: ignore
            return x._variable._data.array.array  # type: ignore

    def __getitem__(self, key):
        return self.get_array()[key]


class _ZarrTiffSlide(NamedTuple):
    zarr: xr.Dataset
    slide: Any

    def close(self):
        for x in self:
            x.close()


def _zarr_tiffslide_opener(fname, **kwargs) -> _ZarrTiffSlide:
    import tiffslide

    slide = tiffslide.TiffSlide(fname, **kwargs)
    zarr: xr.Dataset = xr.open_zarr(
        slide.zarr_group.store,
        consolidated=False,
        mask_and_scale=False,
        chunks=None,  # type: ignore
    )
    return _ZarrTiffSlide(zarr, slide)
