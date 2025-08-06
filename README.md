## Miti360: An integrated dataset combining remote sensing, ground measurements and weather data for improved reforestation monitoring
#### Primer
In the era of artificial intelligence, machine learning combined with remote sensing and ground measurements offers unprecedented opportunities to enhance forest monitoring through faster, more accurate biomass estimation and individual tree analysis. Despite widespread interest, Africa suffers from a shortage of ML-ready forestry datasets, with most major collections—[NEON Crowns](https://doi.org/10.1101/2020.09.08.287839), [Auto Arborist](https://openaccess.thecvf.com/content/CVPR2022/html/Beery_The_Auto_Arborist_Dataset_A_Large-Scale_Benchmark_for_Multiview_Urban_CVPR_2022_paper.html), [ReforesTree](https://doi.org/10.1609/aaai.v36i11.21471), and a [Northern Australia dataset](https://doi.org/10.3390/data8020044)—originating elsewhere. The Miti360 dataset aims to bridge this geographic gap and  is tailored to support data-driven decision-making in establishing and monitoring reforested stands across diverse African landscapes. Existing ML-ready datasets from the Global North have limited relevance in Africa.

#### Contents
The dataset comprises aerial image data (orthophotos and tiles) annotated with bounding boxes for each tree, annotated terrestrial images (single and stereo), tree inventory data (biophysical parameter measurements, GPS coordinates, and species), and historical weather data (precipitation and temperature). These data were collected from a 770-ha reforested section of the Kieni Forest in Kenya between July 2024 and February 2025. 


Below is a tabular summary of the dataset contents:

| #   | Data Category                 | Data Type              | Quantity                                     | Format       |
| --- | ----------------------------- | ---------------------- | -------------------------------------------- | ------------ |
| 1   | Drone Images                  | Orthophoto             | 2                                            | TIF          |
|     |                               | Tiles                  | 844                                          | TIF          |
|     |                               | Tree crown annotations | 24000                                        | JSON         |
|     |                               | Tree crown species     | 1208                                         | CSV          |
|     |                               | Tree species shapefile | 1208                                         | SHP          |
| 2   | Tree ground measurements      | Numeric data           | 1208 (601 trees in 2024 & 607 trees in 2025) | CSV          |
| 3   | Ground based single images    | Images and tree masks  | 1208 (601 trees in 2024 & 607 trees in 2025) | JPEG         |
| 4   | Tree stereo images            | Images and tree masks  | 2416 (601 trees in 2024 & 607 trees in 2025) | JPEG         |
| 5   | Weather data from 40 stations | Time series data       | 8 years daily data                           | API endpoint |

For each tree whose data was recorded during the field survey, there is a single image captured using a smartphone and a pair of images captured with a stereo camera. Other attributes recorded are the location, species, height, crown diameter, and basal diameter. These are captured in CSV files with the following column names:
- `PHONE_IMAGE_FILENAME`: Tree's image taken with a smartphone. Saved in JPG format.
- `LEFT_STEREO_IMAGE_FILENAME`: Left image of the stereo pair. Saved in JPG format.
- `SPECIES`: The species of the sampled tree. Given in standard binomial nomenclature.
- `TH`: Height of the tree in cm.
- `CD`: Crown diameter of the tree in cm.
- `BD`: Basal diameter of the tree in cm.
- `LATITUDE`: GPS latitude of the tree in the WGS84 coordinate frame.
- `LONGITUDE`: GPS longitude of the tree in the WGS84 coordinate frame.

#### Dataset Organisation and Hosting
For information on how the dataset is organised, see this [README text file](README.txt).
The dataset is hosted in two Google Cloud Storage buckets:
- Aerial: Contains aerial images plus associated files.
- Terrestrial: Contains ground-based images plus associated files.

#### Appropriate Usage of Miti360
Miti360 can be used in varied ways to train and assess machine learning models. One useful research angle we have pursued in the past is that of [automating tree inventory](https://doi.org/10.1016/j.softx.2024.101661) using stereoscopic photogrammetry. With recent advances in deep learning and 3D computer vision, the stereoscopic images in Miti360 would be invaluable in developing better techniques for achieving the same goals. Regardless of the ways in which dataset may be used, we believe that all efforts directed towards developing novel techniques for forest monitoring tailored towards our African context will produce the greatest impact.

#### Licensing
Copyright (C) 2025 Centre for Data Science and Artificial Intelligence, DeKUT

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0). 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

#### Contact Information
Cedric Kiplimo: cedric.kiplimo@dkut.ac.ke or dsail-info@dkut.ac.ke