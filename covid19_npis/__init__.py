__version__ = "unknown"
from ._version import __version__

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


from . import test_data
from . import model
from . import plot
from . import utils
from . import data

from .modelParams import ModelParams
