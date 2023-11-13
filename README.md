# TiffSlide-Xarray


A simple integration library between tiffslide and xarray.

## Installation

Install from pypi:

```
pip install tiffslide-xarray
```

# Usage

This library hooks into xarray's extension system as a backend engine. So
it can be used even without importing.

```python
from xarray import open_dataset

slide_level0 = open_dataset("input.svs")
```

The library automatically recoginizes "tiff" and "svs" files. If required,
the "engine" keyword can force usage:

```python
slide_level0 = open_dataset("input.another_extension", engine="tiffslide")
```


Tifflside uses the fsspec and tifffiles packages to open files. Options to these
libraries can be passed using the "storage_options" and "tifffile_options" keyword
arguments.

```python
slide_level0 = open_dataset("s3://input.svs", storage_options={"s3": ... })
```

By default, the level 0 of the the file is read. Other levels can be read by using
the "level" keyword.

```python
slide_level1 = open_dataset("input.svs", level=1)
```

Negative levels are allowed to allow indexing from end of the level array.
```python
slide_level_last = open_dataset("input.svs", level=-1)
```

## Opening All Levels

To open all the levels in the slide, use the "open_all" to return a datatree of the 
slide.

```python
from tiffslide_xarray import open_all_levels

slide = open_all_levels("input.svs")
```

The returned datatree places level0 at the root group, and places subsequent
levels at the f"level{n}" group. 


## Data and MetaData Model


The data for each slide is accessible at "image,"

```python
slide_level0.image
slide_level0["image"]
```

Coordinates for the x, y (and z if it exists) dimensions are added, in units of "px" of the level 0
slide. This makes the cordinates between different levels directly comparable. The library
assumes there are three channels, in the order of (r, g, b). 

```python
>>> slide_level0.x
[0, 1, 2...]
>>> slide_level0.y
[0, 1, 2...]
```

All the metadata from the slide is stored in the dataset attributes. The source file name is
added to the metadata of both the 'image' array and the dataset. If found in the metadata, the microns 
per pixel (mpp) is stored in the "mpp" attributes of the 'x' and 'y' coordinates.

## Lazy Loading

Slides are lazy loaded which makes the initial open very quick, and
loading of small regions is quick (but not cached). Loading of large regions can be slow. 
To manage this, be sure to call "load" on datasets to bring them into memory
if they will be accessed multiple times.

For example, this code will execute two costly reads:

```python
roi = slide_level1.sel(x=slice(10000, 40000), x=slice(5000, 20000))  # select a large ROI

roi2 = 2.0 * roi   # first read
roi2 = 3.0 * roi   # second read
```

Calling "load" on "roi" or "slide_level1" solves this problem.

```python
roi = slide_level1.sel(x=slice(10000, 40000), x=slice(5000, 20000))  # select a large ROI

roi = roi.load() # load the ROI into memory for subsequent processing.
roi2 = 2.0 * roi   # no read
roi2 = 3.0 * roi   # no read
```

# the WSI accessor (experimental)

This package extends xarray with a new accessor, called "wsi," which can be enabled with this import.

```python
import tiffslide_xarray.accessor
```

Eventually, this import will be made automatically, once this api has fully stabilized. The functions of this
accessor can be accessed with the wsi attribute of any xarray dataset or dataarray.

## Coordinate Units in Microns or Pixels

The accessor includes a very lightweight units functionality, based on three methods:

- "set_mpp" sets the microns-per-pixel (stored in the "mpp" attribute) in all coordinates and data_vars with "px" or "um" units.
- "um" optionally sets the microns-per-pixel, but also changes the units of all relevant coordinates/data_vars to "um." 
- "px" optionally sets the microns-per-pixel, but also changes the  units of all relevant coordinates/data_vars to "px." 

All three methods take two arguments:

- "default," which enables setting the mpp to this argument if it does not already exist.
- "override," which sets the mpp to this argument, regardless of whether it is already set.

Using these methods, it is easy ensure a slide has a mpp set, and convert back and forth between px and um coordinates.

```
slide = slide.wsi.um(0.5) # convert coordinates to microns, using a default mpp of 0.5
slide = slide.wsi.px(override=0.5) # convert cordinates to pixels using mpp of 0.5 (regardless of the metadata in the slide).
```

## Regularly-Spaced Grids

There are several tools for regularly spaced, increasing grids. These are grids that are characterized in each
dimension by,

- origin, the point at which the 0,0 pixel lies.
- spacing, the gap between each pixel, which is a constant number.
- size, the number of points in the grid along each number.

This follows closely the semantics of the ITK tooklit. These numbers can be computed for coordinate that is a regularly spaced-grid,
as a dictionary:

```python
slide.wsi.grids
slide.wsi.origin
slide.wsi.spacing
slide.wsi.size
```

It is also possible to compute,

- shift, defined as origin % spacing
- min, the minimum coordinate
- max, the maximum coordinate
- slice, a slice with start=min and stop=max.

The slice can be used with sel to clip objects by the bounding box of another objection. For example, if we have a
region of interest stored in an xarray object (roi),

```python

slide_roi = slide.sel(**roi.wsi.slice)
```

## Converting to PIL

Converting an xarray to a PIL image,

```python
img = slide.image.wsi.pil
```

Keep in mind that displaying the PIL image in jupyter will be very slow for large images.


## Better Defaults for Plot

Data-arrays can plot themsleves as an image with better functionality, and defaults suitable for medical images. In contrast with the 
default imshow of xarray,

1. The aspect ratio is fixed.
2. "yincreasing"  defaults to False


```python
slide.image.wsi.show()
```


# Requesting Feedback


This project currently in alpha to obtain feedback on the API. Please
submit issues or API feature/modification requests to: https://github.com/swamidasslab/tiffslide-xarray.
