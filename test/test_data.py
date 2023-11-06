import xarray as xr

TEST_SLIDE = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"


def test_load():
    x = xr.open_dataset(TEST_SLIDE)

    assert x.attrs["source_file"] == TEST_SLIDE

    assert x.image.attrs["source_file"] == TEST_SLIDE
    assert x.image.attrs["level"] == 0

    assert x.image.dims == ("y", "x", "s")
    assert x.image.shape == (2967, 2220, 3)
    assert x.image.dtype == "uint8"

    assert "x" in x.coords
    assert x.coords["x"].attrs["units"] == "px"

    assert "y" in x.coords
    assert x.coords["y"].attrs["units"] == "px"
