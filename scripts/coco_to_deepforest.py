"""
Thi script converts COCO format JSON annotations to DeepForest DataFrame format.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from shapely.geometry import box


def read_coco_to_deepforest(
    json_path: Union[str, Path],
    output_csv_path: Union[str, Path] = None,
    root_dir: Optional[Union[str, Path]] = None,
    include_geometry: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Read a COCO format JSON file and convert it to DeepForest's expected DataFrame format.
    
    DeepForest expects a DataFrame with columns:
    - image_path: Path/filename to the image
    - xmin, ymin, xmax, ymax: Bounding box coordinates in image space
    - label: Category name (e.g., "tree")
    - geometry: (Optional) Shapely polygon geometry
    
    Args:
        json_path: Path to the COCO JSON file
        root_dir: Optional root directory to prepend to image filenames
        include_geometry: Whether to include shapely geometry polygons (default: True)
    
    Returns:
        pandas DataFrame (or GeoDataFrame if include_geometry=True) with columns:
        image_path, xmin, ymin, xmax, ymax, label, [geometry]
    
    Example:
        >>> df = read_coco_to_deepforest("2024_08_1.json", root_dir="D:/dsail/lacuna_field_work/Aerial/Phase_1/Tiles")
        >>> print(df.head())
    """
    # Load JSON file
    json_path = Path(json_path)
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mappings
    image_map = {img['id']: img['file_name'] for img in coco_data['images']}
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Convert annotations to list of dictionaries
    rows = []
    for ann in coco_data['annotations']:
        # Get image filename
        image_id = ann['image_id']
        file_name = image_map.get(image_id)
        if file_name is None:
            continue  # Skip if image_id not found
        
        # Build image path
        if root_dir:
            image_path = str(Path(root_dir) / file_name)
        else:
            image_path = file_name
        
        # Get category name
        category_id = ann['category_id']
        label = category_map.get(category_id, f"category_{category_id}")
        
        # Convert COCO bbox [x, y, width, height] to [xmin, ymin, xmax, ymax]
        bbox = ann['bbox']
        x, y, width, height = bbox
        xmin = round(x)
        ymin = round(y)
        xmax = round(x + width)
        ymax = round(y + height)
        
        # Create row dictionary
        row = {
            'image_path': image_path,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'label': label.title()
        }
        
        # Add geometry if requested
        if include_geometry:
            row['geometry'] = box(xmin, ymin, xmax, ymax)
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert to GeoDataFrame if geometry is included
    if include_geometry and len(df) > 0:
        try:
            import geopandas as gpd
            df = gpd.GeoDataFrame(df, geometry='geometry')
        except ImportError:
            # If geopandas not available, drop geometry column
            df = df.drop(columns=['geometry'])
            print("Warning: geopandas not available. Geometry column removed.") if verbose else None
    
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"CSV file saved as {os.path.basename(output_csv_path)}") if verbose else None
    
    return df
