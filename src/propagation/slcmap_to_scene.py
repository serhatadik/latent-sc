"""
Convert SLCMap mat file geometry to Sionna RT compatible scene.

This module transforms the raster-based terrain and building data from
SLCMap .mat files into triangulated 3D meshes for use with Sionna ray-tracing.
"""

import os
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import tempfile
import shutil

# Mesh generation (trimesh is generally available, plyfile for export)
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from plyfile import PlyData, PlyElement
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False


class SLCMapToScene:
    """
    Convert SLCMap mat file to Sionna RT scene.
    
    The SLCMap contains:
    - dem: Digital Elevation Model (terrain heights in meters)
    - hybrid_bldg: Building heights (in feet, converted to meters via * 0.3048)
    - cellsize: Grid cell size in meters (typically 30m)
    - axis: UTM coordinate bounds [easting_min, easting_max, northing_min, northing_max]
    
    The output is a Mitsuba 3 XML scene file with:
    - Triangulated terrain mesh
    - Extruded building meshes (grid-based extrusion)
    - ITU radio material assignments
    
    Parameters
    ----------
    slcmap_path : str or Path
        Path to the SLCMap .mat file
    output_dir : str or Path, optional
        Directory to save scene files. If None, uses a temp directory.
    downsample_factor : int, optional
        Factor to downsample the grid (default: 1, no downsampling)
    building_height_threshold : float, optional
        Minimum building height in meters to include (default: 1.0)
    terrain_material : str, optional
        ITU material for terrain (default: "itu_concrete")
    building_wall_material : str, optional
        ITU material for building walls (default: "itu_marble")
    building_roof_material : str, optional
        ITU material for building roofs (default: "itu_metal")
    
    Examples
    --------
    >>> converter = SLCMapToScene("SLCmap_5May2022.mat", output_dir="./scenes")
    >>> scene_path = converter.export_to_mitsuba_xml("slc_scene.xml")
    >>> # Then load in Sionna:
    >>> # from sionna.rt import load_scene
    >>> # scene = load_scene(scene_path)
    """
    
    def __init__(
        self,
        slcmap_path: str,
        output_dir: Optional[str] = None,
        downsample_factor: int = 1,
        building_height_threshold: float = 1.0,
        terrain_material: str = "itu_concrete",
        building_wall_material: str = "itu_marble",
        building_roof_material: str = "itu_metal"
    ):
        if not HAS_TRIMESH:
            raise ImportError("trimesh is required for mesh generation. Install with: pip install trimesh")
        if not HAS_PLYFILE:
            raise ImportError("plyfile is required for mesh export. Install with: pip install plyfile")
            
        self.slcmap_path = Path(slcmap_path)
        if not self.slcmap_path.exists():
            raise FileNotFoundError(f"SLCMap file not found: {slcmap_path}")
            
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="sionna_scene_"))
        self.meshes_dir = self.output_dir / "meshes"
        self.meshes_dir.mkdir(parents=True, exist_ok=True)
        
        self.downsample_factor = downsample_factor
        self.building_height_threshold = building_height_threshold
        self.terrain_material = terrain_material
        self.building_wall_material = building_wall_material
        self.building_roof_material = building_roof_material
        
        # Data holders
        self.dem = None
        self.buildings = None
        self.cellsize = None
        self.axis = None
        self.center_x = None
        self.center_y = None
        
        # Load the map
        self._load_slcmap()
        
    def _load_slcmap(self):
        """Load and parse the SLCMap mat file."""
        print(f"[INFO] Loading SLCMap from: {self.slcmap_path}")
        
        mat_data = sio.loadmat(str(self.slcmap_path))
        map_struct = mat_data['SLC']
        SLC = map_struct[0][0]
        
        # Build column map
        column_map = {name: i for i, name in enumerate(SLC.dtype.names)}
        
        # Extract data
        self.dem = SLC[column_map["dem"]].astype(np.float64)
        hybrid_bldg = SLC[column_map["hybrid_bldg"]].astype(np.float64)
        
        # Convert building heights from feet to meters
        self.buildings = 0.3048 * hybrid_bldg
        
        # Get spatial metadata
        self.cellsize = float(SLC[column_map["cellsize"]][0][0])
        self.axis = SLC[column_map["axis"]].flatten()  # [e_min, e_max, n_min, n_max]
        
        # Apply downsampling
        if self.downsample_factor > 1:
            self.dem = self.dem[::self.downsample_factor, ::self.downsample_factor]
            self.buildings = self.buildings[::self.downsample_factor, ::self.downsample_factor]
            self.cellsize *= self.downsample_factor
            
        # Compute center for scene origin
        self.center_x = (self.axis[0] + self.axis[1]) / 2
        self.center_y = (self.axis[2] + self.axis[3]) / 2
        
        print(f"[INFO] Loaded DEM shape: {self.dem.shape}")
        print(f"[INFO] Cell size: {self.cellsize} m")
        print(f"[INFO] Scene center (UTM): ({self.center_x:.1f}, {self.center_y:.1f})")
        print(f"[INFO] Building cells: {np.sum(self.buildings > self.building_height_threshold)}")
        
    def _get_local_coords(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid indices to local scene coordinates (centered at origin).
        
        Parameters
        ----------
        row : int
            Row index in the grid
        col : int
            Column index in the grid
            
        Returns
        -------
        x, y : float
            Local coordinates in meters
        """
        # UTM coordinates
        easting = self.axis[0] + col * self.cellsize
        northing = self.axis[2] + row * self.cellsize
        
        # Center at origin
        x = easting - self.center_x
        y = northing - self.center_y
        
        return x, y
        
    def generate_terrain_mesh(self) -> str:
        """
        Generate triangulated terrain mesh from DEM.
        
        Creates two triangles per grid cell:
        - Triangle 1: (i,j) → (i+1,j) → (i,j+1)
        - Triangle 2: (i+1,j) → (i+1,j+1) → (i,j+1)
        
        Returns
        -------
        str
            Path to the exported terrain PLY file
        """
        print("[INFO] Generating terrain mesh...")
        
        height, width = self.dem.shape
        
        # Generate vertex coordinates
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Convert to local coordinates
        x = self.axis[0] + cols * self.cellsize - self.center_x
        y = self.axis[2] + rows * self.cellsize - self.center_y
        z = self.dem
        
        # Flatten for vertex array
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        
        vertices = np.column_stack([x_flat, y_flat, z_flat])
        
        # Create vertex data for PLY
        vertex_data = np.array(
            [tuple(v) for v in vertices],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
        
        # Generate faces (triangles)
        # For each cell (i,j) with corners at indices:
        #   (i*width + j), (i*width + j+1), ((i+1)*width + j), ((i+1)*width + j+1)
        faces_list = []
        
        for i in range(height - 1):
            for j in range(width - 1):
                # Corner indices
                idx00 = i * width + j
                idx01 = i * width + (j + 1)
                idx10 = (i + 1) * width + j
                idx11 = (i + 1) * width + (j + 1)
                
                # Triangle 1: (0,0) → (1,0) → (0,1)
                faces_list.append([idx00, idx10, idx01])
                # Triangle 2: (1,0) → (1,1) → (0,1)
                faces_list.append([idx10, idx11, idx01])
                
        faces = np.array(faces_list, dtype=np.int32)
        
        faces_data = np.array(
            [(face,) for face in faces],
            dtype=[('vertex_indices', 'i4', (3,))]
        )
        
        # Export PLY
        terrain_path = self.meshes_dir / "terrain.ply"
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        face_element = PlyElement.describe(faces_data, 'face')
        plydata = PlyData([vertex_element, face_element])
        plydata.write(str(terrain_path))
        
        print(f"[INFO] Terrain mesh: {len(vertices)} vertices, {len(faces)} faces")
        print(f"[INFO] Saved to: {terrain_path}")
        
        return str(terrain_path)
        
    def generate_building_meshes(self) -> Dict[str, str]:
        """
        Generate building meshes using grid-based extrusion.
        
        Each building cell becomes a box from terrain height to terrain + building height.
        Adjacent cells with buildings are merged into connected components for efficiency.
        
        Returns
        -------
        dict
            Mapping of building IDs to PLY file paths
        """
        print("[INFO] Generating building meshes...")
        
        height, width = self.dem.shape
        building_mask = self.buildings > self.building_height_threshold
        
        # Label connected components
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(building_mask)
        
        print(f"[INFO] Found {num_features} building clusters")
        
        building_paths = {}
        
        for building_id in range(1, num_features + 1):
            # Get cells for this building
            cells = np.argwhere(labeled_array == building_id)
            
            if len(cells) == 0:
                continue
                
            # Create meshes for each cell and combine
            cell_meshes = []
            
            for row, col in cells:
                # Get local coordinates for cell corners
                x0 = self.axis[0] + col * self.cellsize - self.center_x
                y0 = self.axis[2] + row * self.cellsize - self.center_y
                x1 = x0 + self.cellsize
                y1 = y0 + self.cellsize
                
                # Heights
                z_ground = self.dem[row, col]
                z_roof = z_ground + self.buildings[row, col]
                
                # Create box mesh for this cell
                box = trimesh.creation.box(
                    extents=[self.cellsize, self.cellsize, self.buildings[row, col]],
                    transform=trimesh.transformations.translation_matrix([
                        (x0 + x1) / 2,
                        (y0 + y1) / 2,
                        (z_ground + z_roof) / 2
                    ])
                )
                cell_meshes.append(box)
                
            # Combine all cells for this building
            if len(cell_meshes) == 1:
                combined = cell_meshes[0]
            else:
                combined = trimesh.util.concatenate(cell_meshes)
                
            # Export PLY
            ply_path = self.meshes_dir / f"building_{building_id}.ply"
            combined.export(str(ply_path), file_type='ply')
            building_paths[f"building_{building_id}"] = str(ply_path)
            
        print(f"[INFO] Generated {len(building_paths)} building meshes")
        
        return building_paths
        
    def export_to_mitsuba_xml(self, filename: str = "scene.xml") -> str:
        """
        Export complete scene to Mitsuba 3 XML format.
        
        Parameters
        ----------
        filename : str
            Name of the XML file (saved in output_dir)
            
        Returns
        -------
        str
            Path to the exported XML file
        """
        print("[INFO] Exporting to Mitsuba XML...")
        
        # Generate all meshes
        terrain_path = self.generate_terrain_mesh()
        building_paths = self.generate_building_meshes()
        
        # Build XML content
        xml_lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<scene version="2.1.0">',
            '',
            '  <!-- Terrain -->',
            f'  <shape type="ply" id="terrain-{self.terrain_material}">',
            f'    <string name="filename" value="meshes/terrain.ply"/>',
            f'    <bsdf type="diffuse" id="{self.terrain_material}">',
            '      <rgb name="reflectance" value="0.5, 0.5, 0.5"/>',
            '    </bsdf>',
            '  </shape>',
            '',
        ]
        
        # Add buildings
        for building_name, ply_path in building_paths.items():
            ply_filename = Path(ply_path).name
            # Alternate between wall and roof materials for variety
            material = self.building_wall_material
            xml_lines.extend([
                f'  <!-- {building_name} -->',
                f'  <shape type="ply" id="{building_name}-{material}">',
                f'    <string name="filename" value="meshes/{ply_filename}"/>',
                f'    <bsdf type="diffuse" id="{material}">',
                '      <rgb name="reflectance" value="0.4, 0.4, 0.4"/>',
                '    </bsdf>',
                '  </shape>',
                '',
            ])
            
        xml_lines.append('</scene>')
        
        # Write XML file
        xml_path = self.output_dir / filename
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(xml_lines))
            
        print(f"[INFO] Scene exported to: {xml_path}")
        print(f"[INFO] Total objects: {1 + len(building_paths)} (1 terrain + {len(building_paths)} buildings)")
        
        return str(xml_path)
        
    def get_scene_bounds(self) -> Dict[str, float]:
        """
        Get the scene bounds in local coordinates.
        
        Returns
        -------
        dict
            Dictionary with min_x, max_x, min_y, max_y, min_z, max_z
        """
        height, width = self.dem.shape
        
        min_x = self.axis[0] - self.center_x
        max_x = self.axis[0] + width * self.cellsize - self.center_x
        min_y = self.axis[2] - self.center_y
        max_y = self.axis[2] + height * self.cellsize - self.center_y
        min_z = float(np.min(self.dem))
        max_z = float(np.max(self.dem + self.buildings))
        
        return {
            'min_x': min_x, 'max_x': max_x,
            'min_y': min_y, 'max_y': max_y,
            'min_z': min_z, 'max_z': max_z
        }


def convert_slcmap_to_sionna_scene(
    slcmap_path: str,
    output_dir: str,
    scene_filename: str = "scene.xml",
    downsample_factor: int = 1,
    **kwargs
) -> str:
    """
    Convenience function to convert SLCMap to Sionna scene.
    
    Parameters
    ----------
    slcmap_path : str
        Path to SLCMap .mat file
    output_dir : str
        Output directory for scene files
    scene_filename : str
        Name of the XML scene file
    downsample_factor : int
        Grid downsampling factor
    **kwargs
        Additional arguments passed to SLCMapToScene
        
    Returns
    -------
    str
        Path to the exported scene XML file
    """
    converter = SLCMapToScene(
        slcmap_path,
        output_dir=output_dir,
        downsample_factor=downsample_factor,
        **kwargs
    )
    return converter.export_to_mitsuba_xml(scene_filename)
