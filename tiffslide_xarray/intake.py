from intake_xarray.netcdf import NetCDFSource
from intake_pattern_catalog.catalog import PatternCatalog
from typing import NamedTuple, Any
import intake
import pandas as pd

class TiffSlideSource(NetCDFSource):
  def __init__(self, xarray_kwargs = {}, **kwargs) -> None:
    xkw = {"engine": "tiffslide"}
    if xarray_kwargs:
      xkw.update(xarray_kwargs)

    super(TiffSlideSource, self).__init__(xarray_kwargs =xkw, **kwargs)


  def read(self):
      """Return a lazy xarray"""
      self._load_metadata()
      return self._ds


    


class TiffSlideGlob(PatternCatalog):
    container = "tiffslide_glob"
    partition_access = None
    name = "tiffslide_glob"
    
    def read(self):
        
        sets: list[dict[str, Any]] = self.get_entry_kwarg_sets()
  
        out = []
        for s in sets:
          s = s.copy()

          entry = self.get_entry(**s)

          s["_intake" ] = entry 

          out.append(s)

        return pd.DataFrame.from_records(out)
        

    

intake.source.discovery.drivers.register_driver(
    "tiffslide_glob",
    TiffSlideGlob,
    clobber=True,
)




intake.source.discovery.drivers.register_driver(
    "tiffslide",
    TiffSlideSource,
    clobber=True,
)
