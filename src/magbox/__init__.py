import importlib.metadata
try:
	__version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
	# When running from a source checkout (not installed as a distribution).
	__version__ = "0.0.0"
__author__="Yutian Wang"

from .llg import llg
from .heff import heff
from .spin import spin

from .llg3 import llg3
from .heff3 import heff3
from .spin3 import spin3
from .boxlib import get_data_type
from .initial import Lattice, Vars

__all__ = ["llg", "heff", "spin", "llg3", "heff3", "spin3", "get_data_type", "Lattice", "Vars"]