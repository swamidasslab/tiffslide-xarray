from __future__ import annotations

"""
We register two xarray accessors:

- grid: for operations on regularly-spaced 1d vectors.
- wsi: for operations domain-specific to whole-slide images.
"""

import xarray as xr
from . import grid
from collections.abc import Mapping
import numpy as np
from typing import Literal
from . import attrs
from typing import TypeVar, Generic


def to_scalar(x):
    return np.ndarray.item(x.data)


xrDataArrayOrSet = TypeVar("xrDataArrayOrSet", xr.Dataset, xr.DataArray)


class CommonAccessor(Generic[xrDataArrayOrSet]):
    _obj: xrDataArrayOrSet

    _namespace: str = "_"

    def __init__(self, xarray_obj: xrDataArrayOrSet):
        self._obj = xarray_obj

    @property
    def _(self) -> xrDataArrayOrSet:
        return self._obj

    def __getitem__(self, key):
        return self._obj[key].wsi

    def data(self):
        return self._obj.data

    @property
    def attrs(self) -> Mapping:
        return attrs.XarrayAttrsProxy(self._obj.attrs)

    @attrs.setter
    def attrs(self, value: Mapping):
        new_attr = {}
        a = attrs.XarrayAttrsProxy(new_attr)
        a.update(value)

        self._obj.attrs.clear()
        self._obj.attrs.update(a.data)

        return attrs.XarrayAttrsProxy(self._obj.attrs)

    def __getattr__(self, key):
        try:
            ret = getattr(self._obj, key)
            try:
                return ret.wsi
            except:
                return ret
        except:
            raise AttributeError

    def _repr_html_(self) -> str:
        n = type(self._obj).__name__
        try:
            type(self._obj).__name__ = n + "." + self._namespace
            return self._obj._repr_html_()
        finally:
            type(self._obj).__name__ = n


class CommonGridAccessor(Generic[xrDataArrayOrSet], CommonAccessor[xrDataArrayOrSet]):
    _namespace = "grid"


@xr.register_dataarray_accessor("grid")
class GridArrayAccessor(CommonGridAccessor[xr.DataArray]):
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
        """
        _summary_

        :return: _description_
        :rtype: int | float
        """
        return self.grid.spacing

    @property
    def shift(self) -> int | float:
        return self.grid.shift

    @property
    def size(self) -> int | None:
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
class GridDatasetAccessor(CommonGridAccessor[xr.Dataset]):
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


class CommonWSIAccessor(Generic[xrDataArrayOrSet], CommonAccessor[xrDataArrayOrSet]):
    _namespace = "wsi"

    def um(self, default=None, override=None, deep=True) -> xrDataArrayOrSet:
        return index_by(self.mpp(default, override, deep)._, "um").wsi

    def px(self, default=None, override=None, deep=True) -> xrDataArrayOrSet:
        return index_by(self.mpp(default, override, deep)._, "px").wsi

    def mpp(self, default=None, override=None, deep=True) -> WSIDatasetAccessor:
        x = self._obj.copy()

        def set_mpp(x, mpp) -> xr.DataArray:
            curr_mpp = x.wsi._mpp
            if not curr_mpp and x.wsi.is_um:
                x.attrs["mpp"] = mpp
                x.wsi.unit = "um"
                return x
            if curr_mpp == mpp:
                return x

            if x.wsi.is_um:
                x = x * mpp / curr_mpp

            x.attrs["mpp"] = mpp
            return x

        mpp = override or infer_mpp(x) or default
        if mpp:
            x = set_mpp(x, mpp)

            if deep:
                vars = list(x.coords)
                if isinstance(x, xr.Dataset):
                    vars += list(x.data_vars)

                for c in vars:
                    cx = x[c]
                    if cx.wsi._mpp or cx.wsi.is_px or cx.wsi.is_um:
                        mpp = override or cx.wsi._mpp or mpp
                        x[c] = set_mpp(cx, mpp)
        return x.wsi

    @property
    def _mpp(self) -> float | None:
        return infer_mpp(self._obj)

    @property
    def unit(self) -> str | None:
        return infer_units(self._obj)

    @unit.setter
    def unit(self, v: str):
        for k in list(self._obj.attrs):
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


@xr.register_dataarray_accessor("wsi")
class WSIArrayAccessor(CommonWSIAccessor[xr.DataArray]):
    def resample(
        self, shift: int | float, spacing: int | float, method="cubic"
    ) -> xr.DataArray:
        raise NotImplemented

    @property
    def pil(self):
        from PIL import Image

        return Image.fromarray(self._obj.data)

    def show(self, *, yincrease=False, figsize=None, aspect=1, **kwargs):
        import matplotlib.pyplot as plt

        if figsize:
            plt.figure(figsize=figsize)

        result = self._obj.plot.imshow(yincrease=yincrease, **kwargs)

        plt.gca().set_aspect(aspect)
        return result


@xr.register_dataset_accessor("wsi")
class WSIDatasetAccessor(
    CommonWSIAccessor[xr.Dataset],
):
    @property
    def is_um(self) -> bool:
        return False

    @property
    def is_px(self) -> bool:
        return False


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

    1. The stored value in x.attr["unit"] or x.attr["units"] (case insensitive)
    2. None.
    """
    # if hasattr(x.data, "units"):  # Triggers reload!!!
    #     return str(x.data.units)

    attr = x.attrs
    for k in attr:
        if k.lower() in {"unit", "units"}:
            return attr[k]

    return None


def index_by(
    slide: xrDataArrayOrSet, unit: Literal["px"] | Literal["um"]
) -> xrDataArrayOrSet:
    """Convert coordinates between pixels (px) and microns (um) units based on the inferred microns-per-pixel (mpp)."""

    assert unit in ["px", "um"]

    def convert(cx: xr.DataArray, mpp) -> xr.DataArray:
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

    for c in vars:
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
