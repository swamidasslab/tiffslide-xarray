import pytest
import xarray as xr
import numpy as np
import tiffslide_xarray
import tiffslide_xarray.accessors  # necessary for accessors


@pytest.fixture
def slide_dataset():
    ny = 7
    nx = 13
    return xr.DataArray(
        np.ones(nx * ny, dtype=np.float32).reshape((ny, nx)),
        dims=("y", "x"),
        coords={
            "x": xr.DataArray(np.arange(nx), dims=("x"), attrs={"unit": "px"}),
            "y": xr.DataArray(np.arange(ny), dims=("y"), attrs={"unit": "px"}),
        },
    ).to_dataset(name="image")


def test_wsi_ipy_repr(slide_dataset):
    S = slide_dataset

    # Datasets are normally displayed as "xarray.Dataset"
    assert "xarray.Dataset<" in S._repr_html_()

    # They are "xarray.Dataset.wsi" in wsi namespace
    assert "xarray.Dataset.wsi" in S.wsi._repr_html_()

    # DataArrays are normally displayed as "xarray.DataArray"
    assert "xarray.DataArray<" in S.image._repr_html_()

    # They are "xarray.DataArray.wsi" in wsi namespace
    assert "xarray.DataArray.wsi" in S.image.wsi._repr_html_()
    assert "xarray.DataArray.wsi" in S.x.wsi._repr_html_()
    assert "xarray.DataArray.wsi" in S.y.wsi._repr_html_()
    assert "xarray.DataArray.wsi" in S.wsi.image._repr_html_()


def test_wsi_namespace(slide_dataset):
    S = slide_dataset

    # "mpp" is not in xarray namespace
    assert not hasattr(S, "mpp")
    assert not hasattr(S.image, "mpp")

    # "mpp" is in wsi namespace
    assert hasattr(S.wsi, "mpp")
    assert hasattr(S.image.wsi, "mpp")

    # method chaining wsi methods stays in wsi namespace
    assert hasattr(S.wsi.mpp().mpp(), "mpp")
    assert hasattr(S.image.wsi.mpp().mpp(), "mpp")
    assert hasattr(S.image.wsi.x, "mpp")
    assert hasattr(S.wsi.x, "mpp")

    # but method chaining xarray methods moves to xarray namespace
    assert not hasattr(S.wsi.coords["x"], "mpp")
    assert not hasattr(S.wsi.sel(), "mpp")

    # the "_" attribute moves from wsi to xarray namespace
    assert not hasattr(S.wsi._, "mpp")
    assert not hasattr(S.image.wsi._, "mpp")

    # xarray variables are accessible in the wsi namespace
    assert S.wsi["image"]._.equals(S["image"])
    assert S.wsi.image.equals(S.image)
    assert S.wsi["image"].equals(S.image)
    assert S.wsi._ is S

    # wsi-space accessors are ducktyped xarrays and numpy arrays
    assert np.all(S.wsi["image"] == S.image)
    assert S.wsi.equals(S)


def test_mpp_coords(slide_dataset):
    S = slide_dataset.copy()

    mpp = 0.5
    x_um = S.x.data * 0.5
    x_px = S.x.data

    # on read, xyz dimensions all have units of "px"
    assert S.x.wsi.unit == "px"
    assert S.y.wsi.unit == "px"

    # calling wsi.px last leads to an x and y-coord  with "px" unit
    assert S.wsi.px(mpp).x.wsi.unit == "px"
    # and x equal to x_px, no matter how many intervening calls of px, mpp, or um
    assert np.allclose(x_px, S.wsi.px(mpp).x._)
    assert np.allclose(x_px, S.wsi.um(mpp).px().x._)
    assert np.allclose(x_px, S.wsi.px(mpp).px().x)
    assert np.allclose(x_px, S.wsi.px(mpp).um().px().x)
    assert np.allclose(x_px, S.wsi.um(mpp).px(mpp).x)
    assert np.allclose(x_px, S.wsi.mpp(mpp).px().x)
    assert np.allclose(x_px, S.wsi.mpp(mpp).um().px().x)

    # calling with new mpp doesn't replace first assigned mpp
    assert np.allclose(x_px * mpp / 2, S.wsi.mpp(mpp / 2).um(mpp).x)
    assert np.allclose(x_px * mpp / 2, S.wsi.px(mpp / 2).um(mpp).x)
    assert np.allclose(x_px * mpp / 2, S.wsi.um(mpp / 2).mpp(mpp).um().x)

    # unless the mpp is set using the "override" keyword
    assert np.allclose(x_um, S.wsi.mpp(mpp * 2).um(override=mpp).x)
    assert np.allclose(x_um, S.wsi.um(mpp * 2).mpp(override=mpp).x)
    assert np.allclose(x_um, S.wsi.px(mpp * 2).um().mpp(override=mpp).x)
    assert np.allclose(x_um, S.wsi.px(mpp * 2).mpp(override=mpp).x.um())

    # calling wsi.um last leads to an x-coord  with "um" unit
    assert S.wsi.um(mpp).x.unit == "um"
    assert S.wsi.um(mpp).y.unit == "um"
    # with the mpp atribute set:
    assert S.wsi.um(mpp).x.attrs["mpp"] == mpp
    assert S.wsi.um(mpp).y.attrs["mpp"] == mpp
    # and x equal to x_um, no matter how many intervening calls of px, mpp, or um
    assert np.allclose(x_um, S.wsi.um(mpp).x)
    assert np.allclose(x_um, S.wsi.um(mpp).um().x)
    assert np.allclose(x_um, S.wsi.px(mpp).um().x)
    assert np.allclose(x_um, S.wsi.px(mpp).px().um().x)
    assert np.allclose(x_um, S.wsi.um(mpp).um(mpp).x)
    assert np.allclose(x_um, S.wsi.mpp(mpp).um().x)
    assert np.allclose(x_um, S.wsi.mpp(mpp).px().um().x)


def test_mpp_datavars(slide_dataset):
    S = slide_dataset.copy()

    mpp = 0.5
    image = S.image.data * 1.0
    # data variables do not have px or um units
    assert S.image.wsi.unit not in ["um", "px"]
    assert not S.image.wsi.is_px
    assert not S.image.wsi.is_um

    # data variables are untouched
    assert np.allclose(image, S.wsi.um(mpp).image)
    assert np.allclose(image, S.wsi.image.um(mpp))
    assert np.allclose(image, S.wsi.px(mpp).image)
    assert np.allclose(image, S.wsi.image.px(mpp))
    assert np.allclose(image, S.wsi.mpp(mpp).image)
    assert np.allclose(image, S.wsi.image.mpp(mpp))

    # unless it has units of um/px, and the units are changed
    S.image.wsi.unit = "um"
    assert S.image.wsi.unit == "um"
    assert S.image.wsi.is_um
    assert np.allclose(image, S.wsi.um(mpp).image)
    assert np.allclose(image / mpp, S.wsi.px(mpp).image)
    assert np.allclose(image, S.wsi.mpp(mpp).image)

    # unless it has units of um/px, and the units are changed
    S.image.wsi.unit = "px"
    assert S.image.wsi.unit == "px"
    assert S.image.wsi.is_px
    assert np.allclose(image * mpp, S.wsi.um(mpp).image)
    assert np.allclose(image * mpp, S.image.wsi.um(mpp))
    assert np.allclose(image, S.wsi.px(mpp).image)
    assert np.allclose(image, S.image.wsi.px(mpp))
    assert np.allclose(image, S.wsi.mpp(mpp).image)
    assert np.allclose(image, S.image.wsi.mpp(mpp))
