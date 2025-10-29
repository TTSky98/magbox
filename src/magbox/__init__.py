import importlib.metadata
__version__=importlib.metadata.version(__name__)
__author__="Yutian Wang"

from .llg import llg
from .heff import heff
from .spin import spin

from .llg3 import llg3
from .heff3 import heff3
from .spin3 import spin3
from .boxlib import get_data_type
from .initial import Lattice, Vars