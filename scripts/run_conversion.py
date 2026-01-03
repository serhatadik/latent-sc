
import sys
import os
from pathlib import Path

# Add project root to path (parent of scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import via src package
try:
    from src.propagation.slcmap_to_scene import SLCMapToScene
except ImportError:
    # Fallback if src is not a package
    sys.path.append(os.path.join(project_root, 'src'))
    from propagation.slcmap_to_scene import SLCMapToScene

def main():
    slcmap_path = "SLCmap_5May2022.mat"
    output_dir = "data/sionna_scenes"
    
    if not os.path.exists(slcmap_path):
        print(f"Error: {slcmap_path} not found")
        return

    print("Running conversion (10x Downsampled to match TIREM)...")
    # Generates 10x downsampled scene to match TIREM's execution resolution
    converter = SLCMapToScene(slcmap_path, output_dir=output_dir, downsample_factor=10)
    
    xml_path = converter.export_to_mitsuba_xml("scene.xml", separate_buildings=True)
    print(f"Conversion done. Scene at: {xml_path}")

if __name__ == "__main__":
    main()
