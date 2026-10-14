"""
Compatibility shims for running pylidc on modern Python + NumPy.

Import this BEFORE pylidc, always. Two independent breakages:

1. configparser.SafeConfigParser
   Deprecated in Python 3.2, REMOVED in 3.12. pylidc calls it when reading
   its config file, so every scan.to_volume() dies without this.
   It was only ever a rename -- SafeConfigParser IS ConfigParser.

2. np.int / np.float / np.bool / np.object ...
   Deprecated in NumPy 1.20, REMOVED in 1.24. pylidc's Contour.to_matrix()
   calls .astype(np.int), which breaks cluster_annotations() -- so no nodules
   are ever found.
   These were plain aliases for the Python builtins; restoring them changes
   no behaviour. np.int is int, exactly.

Both patches are additive: if the attribute already exists, nothing is touched.
"""
import configparser as _cp
import warnings as _w

# -- 1. configparser --------------------------------------------------
if not hasattr(_cp, "SafeConfigParser"):
    _cp.SafeConfigParser = _cp.ConfigParser

# -- 2. numpy scalar aliases ------------------------------------------
import numpy as _np

_ALIASES = {
    "int": int,
    "float": float,
    "bool": bool,
    "object": object,
    "str": str,
    "complex": complex,
    "long": int,
    "unicode": str,
}

with _w.catch_warnings():
    # numpy emits FutureWarning merely for *looking* at these names
    _w.simplefilter("ignore")
    for _name, _builtin in _ALIASES.items():
        try:
            getattr(_np, _name)
        except AttributeError:
            setattr(_np, _name, _builtin)

# pylidc also leans on pkg_resources, which warns loudly on setuptools >= 81.
# The warning is harmless; silence it so real errors stay visible.
_w.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
