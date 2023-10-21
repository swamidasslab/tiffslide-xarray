# TiffSlide-Xarray


A simple integration library between tiffslide and xarray.

## Installation

Install from pypi:

```
pip install tiffslide-xarray
```

## Usage

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

Opening All Levels
==================

To open all the levels in the slide, use the "open_all" to return a datatree of the 
slide.

```python
from tiffslide_xarray import open_all_levels

slide = open_all_levels(input.svs)
```

The returned datatree places level0 at the root group, and places subsequent
levels at the f"level{n}" group. 


Data and MetaData Model
=======================

The data for each slide is accessible at "image,"

```python
slide_level0.image
slide_level0["image]
```

Coordinates for the x, y and c dimensions are added, in units of "pixels" in the level 0
slide. This makes the cordinates between different levels directly compariable. The library
assumes there are three channels, in the order of (r, g, b). 

```python
>>> slide_level0.x
[0, 1, 2...]
>>> slide_level0.y
[0, 1, 2...]
>>> slide_level0.c
['r', 'g', 'b']
```

All the metadata from the slide is stored in the dataset attributes. The source file name is
added to the metadata of both the 'image' array and the dataset. If found in the metadata, the microns 
per pixel (mpp) is stored in the attributes of the 'x' and 'y' coordinates.

Lazy Loading
============

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

Requesting Feedback
===================

This project currently in alpha to obtain feedback on the API. Please
submit issues or API feature/modification requests to: https://github.com/swamidasslab/tiffslide-xarray.
