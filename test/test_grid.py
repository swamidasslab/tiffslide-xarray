import pytest
from hypothesis import given, strategies as st
import numpy as np
from tiffslide_xarray import grid
import os

TEST_WITH_TF  =os.environ.get("TEST_WITH_TF", False)

@given(
    origin=st.floats(allow_nan=False, allow_infinity=False),
    spacing=st.floats(0.0001, 100000, allow_nan=False, allow_infinity=False),
    size=st.integers(0),
)
def test_identity_downsample(origin, spacing, size):
    d = grid.DownSamplerShape(1, 0, 0)  # identity

    g = grid.RegularGrid(origin, spacing, size)
    out = g.downsample(d)

    assert out == g

    g = grid.RegularGrid(origin, spacing, None)
    out = g.downsample(d)

    assert out == g


@given(
    origin=st.floats(-10000, 10000, allow_nan=False, allow_infinity=False),
    spacing=st.floats(0.0001, 100000, allow_nan=False, allow_infinity=False),
    size=st.integers(0, 4000),
    downsample=st.integers(1, 100),
)
def test_no_padding_downsample(origin, spacing, size, downsample):
    d = grid.DownSamplerShape(downsample, 0, 0)
    first = (np.arange(downsample) * spacing + origin).mean()
    length = size // downsample

    g = grid.RegularGrid(origin, spacing, size)
    out = g.downsample(d)

    assert np.allclose(first, out.origin)
    assert out.size is not None
    assert np.allclose(length, out.size)

    g = grid.RegularGrid(origin, spacing, None)
    out = g.downsample(d)

    assert np.allclose(first, out.origin)
    assert out.size is None


def test_default_identity():
    assert grid.DownSamplerShape() == grid.DownSamplerShape(1, 0, 0)


def test_scalar_math():
    assert grid.RegularGrid() + 2 == grid.RegularGrid(origin=2)
    assert 3 + grid.RegularGrid() == grid.RegularGrid(origin=3)
    assert grid.RegularGrid(origin=1) * 2 == grid.RegularGrid(origin=2, spacing=2)
    assert 3 * grid.RegularGrid(origin=1) == grid.RegularGrid(origin=3, spacing=3)
    assert grid.RegularGrid(origin=2) / 2 == grid.RegularGrid(origin=1, spacing=1 / 2)
    assert grid.RegularGrid(origin=3) / 3 == grid.RegularGrid(origin=1, spacing=1 / 3)


def test_slicing():
    assert grid.RegularGrid()[3:] == grid.RegularGrid(origin=3)
    assert grid.RegularGrid(spacing=3)[3:] == grid.RegularGrid(origin=9, spacing=3)
    assert grid.RegularGrid(spacing=3, size=20)[3:] == grid.RegularGrid(
        origin=9, spacing=3, size=17
    )
    assert grid.RegularGrid(spacing=2, size=20)[:-3] == grid.RegularGrid(
        spacing=2, size=17
    )
    assert grid.RegularGrid()[::2] == grid.RegularGrid(spacing=2)
    assert grid.RegularGrid()[1::2] == grid.RegularGrid(origin=1, spacing=2)


@given(
    origin=st.floats(-10000, 10000),
    spacing=st.floats(0.0001, 10),
    size=st.integers(0, 1000),
    downsample=st.integers(1, 100),
    padding=st.integers(0, 100),
    odd=st.integers(0, 1),
)
def test_padding_downsample(origin, spacing, size, downsample, padding, odd):
    g = grid.RegularGrid(origin, spacing, size)
    d = grid.DownSamplerShape(downsample, padding, odd)

    padding_odd = padding + odd * 0.5

    first = (np.arange(downsample).mean() + padding_odd) * spacing + origin
    length = max(size - padding_odd * 2, 0) // downsample
    skips = spacing * downsample

    out = g.downsample(d)

    o = out.origin
    assert np.allclose(first, o)
    assert out.size is not None
    assert np.allclose(length, out.size)
    assert np.allclose(skips, out.spacing)


@st.composite
def keras_model_1D(draw):
    import tensorflow.keras as K

    i = K.Input((None, 3))
    x = i

    for _ in range(draw(st.integers(1, 3))):
        x = K.layers.Conv1D(
            1,
            kernel_size=draw(st.integers(1, 6)),
            strides=draw(st.integers(1, 3)),
            padding=draw(st.sampled_from(["same", "valid"])),
        )(x)

    return K.Model(i, x)

@pytest.mark.skipif(not TEST_WITH_KERAS, reason="TEST_WITH_TF env variable not set.")
@given(model=keras_model_1D())
def test_keras_analysis(model):
    import tensorflow.keras as K

    def size_function(x):
        try:
            s = model.compute_output_shape((None, x, None))
            return s[1]
        except ValueError:
            return 0

    downsampler = grid.DownSamplerShape.from_function(size_function)

    for x in range(0, 100):
        keras_out_size = size_function(x)
        g = grid.RegularGrid(size=x)

        new_grid = g.downsample(downsampler)

        assert keras_out_size == new_grid.size
