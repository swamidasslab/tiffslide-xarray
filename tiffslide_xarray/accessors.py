from __future__ import annotations

"""
We register two xarray accessors:

- grid: for operations on regularly-spaced 1d vectors.
- wsi: for operations domain-specific to whole-slide images.
"""

import xarray as xr
from . import grid
import numpy as np
from typing import Literal, overload



def to_scalar(x):
    return np.ndarray.item(x.data)


@xr.register_dataarray_accessor("grid")
class GridArrayAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self._check: bool | None = None
        self._grid: grid.RegularGrid | None = None

    @property
    def grid(self) -> grid.RegularGrid:
        if self._grid:
            return self._grid
        x = self._obj
        assert x.grid.check, "Not a regularly spaced 1D vector."
        self._grid = grid.RegularGrid.from_coord(x.data)
        return self._grid

    def clear(self):
        """Clear cache."""
        self._check = self._grid = None

    def _check_regular_spacing(self) -> bool:
        x = self._obj.data
        if len(x.shape) != 1:
            return False

        if len(x) < 2:
            return True

        deltas = x[:-1] - x[1:]
        return np.allclose(deltas[:-1], deltas[1:])

    @property
    def check(self) -> bool:
        """Returns True only if this is a regularly spaced 1D vector. Result is cached, and will not update if values of array are altered."""
        if self._check:
            return self._check
        self._check = self._check_regular_spacing()
        return self._check

    @property
    def origin(self) -> int | float:
        return self.min

    @property
    def spacing(self) -> int | float:
        return self.grid.spacing

    @property
    def shift(self) -> int | float:
        return self.grid.shift

    @property
    def size(self) -> int | float:
        return self.grid.size

    @property
    def min(self) -> int | float:
        assert len(self._obj.dims) == 1
        return to_scalar(self._obj[0])

    @property
    def max(self) -> int | float:
        assert len(self._obj.dims) == 1
        return to_scalar(self._obj[-1])

    @property
    def range(self) -> int | float:
        return self.max - self.min  # type: ignore


def _safe_coord_map(func, x: xr.Dataset):
    out = {}
    for k in x.coords:
        try:
            out[k] = func(x.coords[k])
        except AssertionError:
            pass
    return out


@xr.register_dataset_accessor("grid")
class GridDatasetAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    @property
    def grid(self) -> dict[str, grid.RegularGrid]:
        return _safe_coord_map(lambda x: x.grid.grid, self._obj)

    @property
    def origin(self) -> dict[str, int | float]:
        return _safe_coord_map(lambda x: x.grid.origin, self._obj)

    @property
    def spacing(self) -> dict[str, int | float]:
        return _safe_coord_map(lambda x: x.grid.spacing, self._obj)

    @property
    def shift(self) -> dict[str, int | float]:
        return _safe_coord_map(lambda x: x.grid.shift, self._obj)

    @property
    def size(self) -> dict[str, int | float]:
        return _safe_coord_map(lambda x: x.grid.size, self._obj)

    @property
    def min(self) -> dict[str, int | float]:
        return _safe_coord_map(lambda x: x.grid.min, self._obj)

    @property
    def max(self) -> dict[str, int | float]:
        return _safe_coord_map(lambda x: x.grid.max, self._obj)

    @property
    def range(self) -> dict[str, int | float]:
        return _safe_coord_map(lambda x: x.grid.max, self._obj)


@xr.register_dataarray_accessor("wsi")
class TiffArrayAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    @property
    def um(self) -> xr.DataArray:
        return index_by(self._obj, "um")

    @property
    def px(self) -> xr.DataArray:
        return index_by(self._obj, "px")

    def resample(
        self, shift: int | float, spacing: int | float, method="cubic"
    ) -> xr.DataArray:
        raise NotImplemented

    @property
    def unit(self) -> str | None:
        return infer_units(self._obj)

    @unit.setter
    def unit(self, v: str):
        for k in self._obj.attrs:
            if k.lower() in {"unit", "units"}:
                del self._obj.attrs[k]

        self._obj.attrs["units"] = v

    @property
    def is_px(self) -> bool:
        u = str(self._obj.wsi.unit)
        return u.strip().lower() in {"px", "pixel", "pixels"}

    @property
    def is_um(self) -> bool:
        u = str(self._obj.wsi.unit)
        return u.strip().lower() in {
            "um",
            "micron",
            "microns",
            "micrometer",
            "micrometers",
        }

    def mpp(self, default=None, override=None, deep=True) -> TiffArrayAccessor:
        x = self._obj.copy()
        mpp = override or infer_mpp(x) or default
        if mpp:
            if x.wsi._mpp or x.wsi.is_px or x.wsi.is_um:
                x.attrs["mpp"] = mpp

            if deep:
                for c in list(x.coords):
                    c = x.coords[c]
                    if c.wsi._mpp or c.wsi.is_px or c.wsi.is_um:
                        c.attrs["mpp"] = override or c.wsi._mpp or mpp
        return x.wsi

    @property
    def _mpp(self) -> float | None:
        return infer_mpp(self._obj)

    @property
    def pil(self):
        from PIL import Image

        return Image.fromarray(self._obj.data)

    def show(self, *, yincrease=False, **kwargs):
        result = self._obj.plot.imshow(yincrease=yincrease, **kwargs)
        import matplotlib.pyplot as plt

        plt.gca().set_aspect(1)
        return result


@xr.register_dataset_accessor("wsi")
class TiffDatasetAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    @property
    def um(self) -> xr.Dataset:
        return index_by(self._obj, "um")

    @property
    def px(self) -> xr.Dataset:
        return index_by(self._obj, "px")

    def mpp(self, default=None, override=None, deep=True) -> TiffDatasetAccessor:
        x = self._obj.copy()
        mpp = override or infer_mpp(x) or default
        if mpp:
            x.attrs["mpp"] = mpp

            if deep:
                for c in list(x.coords):
                    c = x.coords[c]
                    if c.wsi._mpp or c.wsi.is_px or c.wsi.is_um:
                        c.attrs["mpp"] = override or c.wsi._mpp or mpp
                for c in list(x.data_vars):
                    c = x[c]
                    if c.wsi._mpp in c.attrs or c.wsi.is_px or c.wsi.is_um:
                        c.attrs["mpp"] = override or c.wsi._mpp or mpp

        return x.wsi

    @property
    def _mpp(self) -> float | None:
        return infer_mpp(self._obj)

    @property
    def unit(self) -> str | None:
        return infer_units(self._obj)

    @property
    def is_px(self) -> bool:
        u = str(self._obj.wsi.unit)
        return u.strip().lower() in {"px", "pixel", "pixels"}

    @property
    def is_um(self) -> bool:
        u = str(self._obj.wsi.unit)
        return u.strip().lower() in {
            "um",
            "micron",
            "microns",
            "micrometer",
            "micrometers",
        }


def infer_mpp(x: xr.Dataset | xr.DataArray) -> float | None:
    """Returns the units of the xarray, inferred from (by precedence):

    1. The stored value in x.attrs["mpp"]
    2. The stored value in x.attrs["tiffslide.mpp"]
    3. None.
    """
    if "mpp" in x.attrs:
        return x.attrs["mpp"]
    if "tiffslide.mpp" in x.attrs:
        return x.attrs["tiffslide.mpp"]


def infer_units(x: xr.Dataset | xr.DataArray) -> str | None:
    """Returns the units of the xarray, inferred from (by precedence):

    1. The "x.data.units" attribute, where pint stores units.
    2. The stored value in x.attr["unit"] or x.attr["units"] (case insensitive)
    3. None.
    """
    if hasattr(x.data, "units"):
        return str(x.data.units)

    attr = x.attrs
    for k in attr:
        if k.lower() in {"unit", "units"}:
            return attr[k]

    return None


@overload
def index_by(slide: xr.Dataset, unit: Literal["px"] | Literal["um"]) -> xr.Dataset:
    ...


@overload
def index_by(slide: xr.DataArray, unit: Literal["px"] | Literal["um"]) -> xr.DataArray:
    ...


def index_by(
    slide: xr.Dataset | xr.DataArray, unit: Literal["px"] | Literal["um"]
) -> xr.Dataset | xr.DataArray:
    """Convert coordinates between pixels (px) and microns (um) units based on the inferred microns-per-pixel (mpp)."""

    assert unit in ["px", "um"]

    def convert(cx, mpp):
        if unit == "um":
            if cx.wsi.is_px:
                cx = cx * mpp
                cx.wsi.unit = unit
                cx.attrs["mpp"] = mpp

        elif unit == "px":
            if cx.wsi.is_um:
                cx = cx / mpp
                cx.wsi.unit = unit
                cx.attrs["mpp"] = mpp

        return cx

    slide = slide.copy()
    mpp_found = False

    vars = list(slide.coords)

    if isinstance(slide, xr.Dataset):
        vars += list(slide.data_vars)
    elif slide.wsi._mpp:
        slide = convert(slide, slide.wsi._mpp)

    for c in list(slide.coords):
        cx = slide[c]

        # continue if can't infer mpp
        mpp = cx.wsi._mpp or slide.wsi._mpp

        if not mpp:
            continue
        mpp_found = True

        slide[c] = convert(cx, mpp)

    if not mpp_found:
        raise ValueError("Could not infer mpp on any coordinates.")

    return slide
