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
            
        # FIX: Handle NoData values (0.0 elevation) robustly
        # Use Nearest Neighbor inpainting to fill holes smoothly
        invalid_mask = self.dem <= 1.0 # Assuming site > 1m
        num_invalid = np.sum(invalid_mask)
        
        if num_invalid > 0:
            print(f"[WARNING] Found {num_invalid} pixels with invalid elevation (<=1m). Inpainting with Nearest Neighbors...")
            try:
                from scipy.ndimage import distance_transform_edt
                # Compute indices of nearest valid pixel (value 0 in mask)
                # invalid_mask is 1 where we want to fill, 0 where data is valid.
                # distance_transform_edt finds nearest background (0) pixel.
                indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
                
                # Use the indices to fetch values from self.dem
                # This effectively replaces every pixel with its nearest valid neighbor 
                # (Valid pixels replace themselves as they are their own nearest neighbor)
                self.dem = self.dem[tuple(indices)]
                print("[INFO] Inpainting complete.")
                
            except ImportError:
                print("[ERROR] scipy.ndimage required for inpainting. Falling back to global min fill.")
                valid_mask = ~invalid_mask
                if np.any(valid_mask):
                    min_valid = np.min(self.dem[valid_mask])
                    self.dem[invalid_mask] = min_valid
        
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
        
    def generate_terrain_mesh(self, include_buildings: bool = True) -> str:
        """
        Generate triangulated terrain mesh from elevation data.
        
        By default, this generates a COMBINED elevation mesh (DEM + building heights)
        which matches exactly what TIREM uses for its terrain model:
            elev_map = dem + 0.3048 * hybrid_bldg
        
        Creates two triangles per grid cell for a smooth heightfield surface.
        
        Parameters
        ----------
        include_buildings : bool, optional
            If True (default), include building heights in elevation.
            This produces a mesh matching TIREM's terrain representation.
            If False, only use raw DEM terrain.
        
        Returns
        -------
        str
            Path to the exported terrain PLY file
        """
        if include_buildings:
            print("[INFO] Generating combined elevation mesh (DEM + buildings)...")
            # Use combined elevation - EXACTLY matching TIREM's approach
            elevation = self.dem + self.buildings
            mesh_name = "terrain_combined.ply"
        else:
            print("[INFO] Generating terrain mesh (DEM only)...")
            elevation = self.dem
            mesh_name = "terrain.ply"
        
        height, width = elevation.shape
        
        # Generate vertex coordinates
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Convert to local coordinates
        x = self.axis[0] + cols * self.cellsize - self.center_x
        y = self.axis[2] + rows * self.cellsize - self.center_y
        z = elevation  # Use combined or DEM-only elevation
        
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
        terrain_path = self.meshes_dir / mesh_name
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        face_element = PlyElement.describe(faces_data, 'face')
        plydata = PlyData([vertex_element, face_element])
        plydata.write(str(terrain_path))
        
        z_range = f"[{z.min():.1f}, {z.max():.1f}]"
        print(f"[INFO] Elevation mesh: {len(vertices)} vertices, {len(faces)} faces")
        print(f"[INFO] Elevation range: {z_range} m")
        print(f"[INFO] Saved to: {terrain_path}")
        
        return str(terrain_path)
        
    def generate_building_meshes(self) -> Dict[str, str]:
        """
        Generate building meshes by extracting components from the combined heightfield.
        
        This method identifies connected components of building pixels and generates
        a mesh for each component using the EXACT geometry of the combined elevation
        grid (DEM + Buildings). This ensures perfect alignment with TIREM's view
        of the data, while separating buildings into individual objects for
        material assignment.
        
        The "walls" of the buildings are formed by the slope between building pixels
        and the surrounding terrain pixels (which matches the raster interpolation).
        
        Returns
        -------
        dict
            Mapping of building IDs to PLY file paths
        """
        print("[INFO] Generating building meshes (heightfield-based)...")
        
        from scipy import ndimage
        
        # Identify building pixels
        building_mask = self.buildings > self.building_height_threshold
        
        # Label connected components
        labeled_array, num_features = ndimage.label(building_mask)
        # Get bounding boxes for efficiency
        objects = ndimage.find_objects(labeled_array)
        
        print(f"[INFO] Found {num_features} building clusters")
        
        building_paths = {}
        
        for building_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
                
            # Expand slice by 1 pixel to include the "ramp" (wall) down to terrain
            # Be careful with boundaries
            y_slice, x_slice = slices
            
            y_start = max(0, y_slice.start - 1)
            y_stop = min(self.dem.shape[0], y_slice.stop + 1)
            x_start = max(0, x_slice.start - 1)
            x_stop = min(self.dem.shape[1], x_slice.stop + 1)
            
            # Extract sub-grids
            sub_dem = self.dem[y_start:y_stop, x_start:x_stop]
            sub_bldg = self.buildings[y_start:y_stop, x_start:x_stop]
            sub_mask = (labeled_array[y_start:y_stop, x_start:x_stop] == building_id)
            
            # Use combined elevation for vertices
            sub_elevation = sub_dem + sub_bldg
            
            h, w = sub_elevation.shape
            if h < 2 or w < 2:
                continue
                
            # Create local vertices
            # Note: We need to use absolute coordinates to match scene
            rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            
            # Map local implementation indices to global indices
            global_rows = rows + y_start
            global_cols = cols + x_start
            
            # Convert to scene coordinates
            x = self.axis[0] + global_cols * self.cellsize - self.center_x
            y = self.axis[2] + global_rows * self.cellsize - self.center_y
            z = sub_elevation
            
            vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
            
            vertex_data = np.array(
                [tuple(v) for v in vertices],
                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            )
            
            # Generate faces
            # Only keep faces that involve at least one building pixel
            # A face (quad) is formed by (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            # Check mask at these corners
            
            mask_flat = sub_mask.flatten()
            
            faces_list = []
            
            # Vectorized face generation for this patch
            # Top-left indices of quads
            r_idx = np.arange(h - 1)
            c_idx = np.arange(w - 1)
            R, C = np.meshgrid(r_idx, c_idx, indexing='ij')
            
            idx00 = R * w + C
            idx01 = R * w + (C + 1)
            idx10 = (R + 1) * w + C
            idx11 = (R + 1) * w + (C + 1)
            
            # Check if any vertex in the quad belongs to the building
            # We treat the quad as part of the building if ANY vertex is part of the component
            # This ensures the "ramp" is included.
            v00_mask = mask_flat[idx00]
            v01_mask = mask_flat[idx01]
            v10_mask = mask_flat[idx10]
            v11_mask = mask_flat[idx11]
            
            quad_mask = v00_mask | v01_mask | v10_mask | v11_mask
            
            # Extract valid quads
            valid_idx00 = idx00[quad_mask]
            valid_idx01 = idx01[quad_mask]
            valid_idx10 = idx10[quad_mask]
            valid_idx11 = idx11[quad_mask]
            
            # Create two triangles per quad
            # Tri 1: 00, 10, 01
            t1 = np.column_stack([valid_idx00, valid_idx10, valid_idx01])
            # Tri 2: 10, 11, 01
            t2 = np.column_stack([valid_idx10, valid_idx11, valid_idx01])
            
            faces = np.vstack([t1, t2])
            
            if len(faces) == 0:
                continue
                
            faces_data = np.array(
                [(face,) for face in faces],
                dtype=[('vertex_indices', 'i4', (3,))]
            )
            
            # Export
            ply_path = self.meshes_dir / f"building_{building_id}.ply"
            
            vertex_element = PlyElement.describe(vertex_data, 'vertex')
            face_element = PlyElement.describe(faces_data, 'face')
            
            # Because PlyData writes are slow for many small files, we trust the OS buffer
            plydata = PlyData([vertex_element, face_element])
            plydata.write(str(ply_path))
            
            building_paths[f"building_{building_id}"] = str(ply_path)
            
        print(f"[INFO] Generated {len(building_paths)} building meshes")
        return building_paths
        
    def export_to_mitsuba_xml(self, filename: str = "scene.xml", separate_buildings: bool = True) -> str:
        """
        Export complete scene to Mitsuba 3 XML format.
        
        Parameters
        ----------
        filename : str
            Name of the XML file (saved in output_dir)
        separate_buildings : bool, optional
            If True (default), generate separate meshes for terrain and buildings.
            This uses the heightfield extraction method to preserve exact geometry
            while allowing different materials.
            If False, generate a single combined elevation mesh (one object).
            
        Returns
        -------
        str
            Path to the exported XML file
        """
        print("[INFO] Exporting to Mitsuba XML...")
        
        if separate_buildings:
            # Separate building and terrain extraction
            # Terrain: Pure DEM (flattened ground under buildings is fine, 
            # buildings will sit on top/cover it)
            terrain_path = self.generate_terrain_mesh(include_buildings=False)
            building_paths = self.generate_building_meshes()
        else:
            # Single combined elevation mesh
            terrain_path = self.generate_terrain_mesh(include_buildings=True)
            building_paths = {}
        
        # Get terrain mesh filename
        terrain_filename = Path(terrain_path).name
        
        # Build XML content
        # Define materials globally at the top
        xml_lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<scene version="2.1.0">',
            '',
            '  <!-- Material Definitions -->',
            f'  <bsdf type="diffuse" id="{self.terrain_material}">',
            '    <rgb name="reflectance" value="0.5, 0.5, 0.5"/>',
            '  </bsdf>',
            '',
            f'  <bsdf type="diffuse" id="{self.building_wall_material}">',
            '    <rgb name="reflectance" value="0.4, 0.4, 0.4"/>',
            '  </bsdf>',
            '',
            '  <!-- Components -->',
            '  <!-- Terrain -->',
            f'  <shape type="ply" id="terrain">',
            f'    <string name="filename" value="meshes/{terrain_filename}"/>',
            f'    <ref id="{self.terrain_material}"/>',
            '  </shape>',
            '',
        ]
        
        # Add separate buildings only if requested
        if separate_buildings:
            for building_name, ply_path in building_paths.items():
                ply_filename = Path(ply_path).name
                material = self.building_wall_material
                
                # Combine building name with material for unique shape ID (optional, but good practice)
                shape_id = f"{building_name}"
                
                xml_lines.extend([
                    f'  <!-- {building_name} -->',
                    f'  <shape type="ply" id="{shape_id}">',
                    f'    <string name="filename" value="meshes/{ply_filename}"/>',
                    f'    <ref id="{material}"/>',
                    '  </shape>',
                    '',
                ])
            
        xml_lines.append('</scene>')
        
        # Write XML file
        xml_path = self.output_dir / filename
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(xml_lines))
            
        n_objects = 1 + len(building_paths)
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
