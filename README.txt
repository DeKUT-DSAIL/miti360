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
│   │   │   ├──bounding_boxes/
│   │   │   │   └── 2024_08_annotations.shp
│   │   │   │   └── ...
│   │   │   ├──tree_attributes/
│   │   │   │   └── tree_attributes_ex_1.shp
│   │   │   │   └── ...
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
│   │   │   ├──bounding_boxes/
│   │   │   │   └── 2025_02_annotations.shp
│   │   │   │   └── ...
│   │   │   ├──tree_attributes/
│   │   │   │   └── tree_attributes_ex_2.shp
│   │   │   │   └── ...
│   │   ├── Tiles/
│   │   │   └── xxx.tif
│   │   │   └── ...
│   │   ├── Annotations/
│   │   │   └── xxx.json
│   │   │   └── ...
│   

Terrestrial/
├── Phase_1/
│   ├── config/
│   │   └── single_1.yml
│   │   └── stereo_1.yml
│
│   ├── images/
│   │   ├── phone/
│   │   │   └── xxx.jpg
│   │   │   └── ...
│
│   │   ├── stereo/
│   │   │   └── xxx_left.jpg
│   │   │   └── xxx_right.jpg
│   │   │   └── ...
│
│   ├── masks/
│   │   ├── phone/
│   │   │   └── xxx_mask.jpg
│   │   │   └── ...
│
│   │   ├── stereo/
│   │   │   └── xxx_left_mask.jpg
│   │   │   └── xxx_right_mask.jpg
│   │   │   └── ...
│
│   ├── metadata/
│   │   └── tree_attributes_phase_1.csv
│
├── Phase_2/
│   ├── ...



=========================================================================
Naming Convention
=========================================================================
1. Tiles: year_month_{tile_id}.tif
2. Single images: year_month_{plot_id}_phone_{tree_id}.jpg
3. Single image masks: year_month_{plot_id}_phone_{tree_id}_mask.jpg
4. Stereo images: year_month_{plot_id}_stereo_{tree_id}_{left/right}.jpg
5. Stereo image masks: year_month_{plot_id}_stereo_{tree_id}_left_mask.jpg
6. Camera configuration files: {single/stereo}_{phase_id}.yml
7. Tree attribute files: tree_attributes_phase_{phase_id}.csv


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


=========================================================================
NOTE on Masks
=========================================================================
For every image inside the images/phone or images/stereo folders, the corresponding mask is located in the masks/phone or masks/stereo folder.
The only difference between the filenames is in the suffix "_mask" appended to the image's filename before the extention. 
For example, the mask of image 2024_07_01_phone_001.jpg is 2024_07_01_phone_001_mask.jpg, and so on. 