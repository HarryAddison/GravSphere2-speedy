"""GravSphere2: higher-order Jeans modelling of spherical stellar systems.

GravSphere2 mass-models galaxies and star clusters by solving the spherical
second- and fourth-order Jeans equations without binning the data.  See
:mod:`gravsphere2.config` for the input-file schema and
:class:`gravsphere2.likelihood.GravSphere2Model` for the likelihood.
"""

__version__ = "2.0.0"

from .config import Config, ConfigError
from .data import Dataset, load_dataset
from .likelihood import GravSphere2Model

__all__ = ["Config", "ConfigError", "Dataset", "GravSphere2Model", "load_dataset", "__version__"]
