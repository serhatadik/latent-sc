from .base import PropagationModel
from .log_distance import LogDistanceModel
from .tirem_wrapper import TiremModel

# Sionna model is optional (requires sionna, yaml, etc.)
try:
    from .sionna_wrapper import SionnaModel
except (ImportError, ModuleNotFoundError):
    SionnaModel = None

# Scene conversion utilities (requires trimesh, plyfile)
try:
    from .slcmap_to_scene import SLCMapToScene, convert_slcmap_to_sionna_scene
except (ImportError, ModuleNotFoundError):
    SLCMapToScene = None
    convert_slcmap_to_sionna_scene = None
