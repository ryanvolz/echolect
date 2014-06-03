from .calc import *
from .coding import *
from .indexing import *
from .plotting import *
from .subsectime import *

import warnings
try:
    import glumpy
except ImportError:
    warnings.warn('glumpy not available, video support is not loaded',
                  ImportWarning)
else:
    from .video import *
    del glumpy
finally:
    del warnings
