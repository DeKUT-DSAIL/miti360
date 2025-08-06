Miti360: An integrated dataset combining remote sensing, ground measurements and weather data for improved reforestation monitoring


=========================================================================
Folder Structure
=========================================================================
The dataset has been organised into two folders -  one for aerial images plus associated files and the other terrestrial images plus associated files. Each of the two folders has been zipped and stored in
a Google Cloud Storage bucket. 

The directory structure of the two folders is as follows:

Aerial/
│   ├──Phase_1/
│   │   ├── Orthophoto/
│   │   │   └── xxx.tif
│   │   │   ├──shapefiles/
│   │   │   │   └── bounding_boxes.tif
│   │   │   │   └── tree_attributes.tif
│   │   ├── Tiles/
│   │   │   └── xxx.tif
│   │   │   └── ...
│   │   ├── Annotations/
│   │   │   └── xxx.json
│   │   │   └── ...
│   
│   ├──Phase_2/
│   │   ├── Orthophoto/
│   │   │   └── xxx.tif
│   │   │   ├──shapefiles/
│   │   │   │   └── bounding_boxes.tif
│   │   │   │   └── tree_attributes.tif
│   │   ├── Tiles/
│   │   │   └── xxx.tif
│   │   │   └── ...
│   │   ├── Annotations/
│   │   │   └── xxx.json
│   │   │   └── ...
│   

Terrestrial/
├── Phase_1/
│   ├── images/
│   │   ├── phone/
│           └── xxx.jpg
│           └── ...
│   │   ├── stereo/
│           └── xxx_left.jpg
│           └── xxx_right.jpg
│           └── ...
│   │   ├── config/
│           └── single_1.yml
│           └── stereo_1.yml
│
│   ├── masks/
│   │   ├── phone/
│           └── xxx_mask.jpg
│           └── ...
│   │   ├── stereo/
│           └── xxx_left_mask.jpg
│           └── xxx_right_mask.jpg
│           └── ...
│   │   ├── config/
│           └── single_2.yml
│           └── stereo_2.yml
│
├── Phase_2/
│   ├── ...



=========================================================================
Naming Convention
=========================================================================
1. Tiles: year_month_{tile_id}.tif
2. Single images: year_month_{plot_id}_phone_{tree_id}.jpg
3. Stereo images: year_month_{plot_id}_stereo_{tree_id}_{left/right}.jpg
4. Camera configuration files: {single/stereo}_{phase_id}.yml


=========================================================================
Meaning of fields:
=========================================================================
1. year: Year when image was captured or file created
2. month: Month when image was captured or file created
3. tile_id: A unique three-digit integer indicating the position of the tile in the sequence
4. plot_id: A unique two-digit integer identifying the specific plot in the sampling design where the picture was taken
5. tree_id: A unique  three-digit integer indicating the position of the tree in the sequence of all trees sampled during one phase of the data collection
6. left/right: Identifies whether a stereo image is the left or right one in the pair
7. single/stereo: Identifies whether the camera calibration file is for a single camera or a stereo camera
8. phase_id: Identifies the camera calibration file is for the stereo camera during which phase of data collection e.g., 1 or 2. 