# Folders
```
└── assets : Has resources such as GroundTruth, Bitrates, Plots etc.
|   ├── Bitrates
|   ├── F2s
|   ├── GroundTruths
|   ├── GroundTruths_TileLevel
|   ├── Plots
|   ├── Runtimes
|   |   ├── On Nano
|   |   ├── On RPi4
|   |
|   ├── labels
|   ├── rates.pkl
|
└── extras: Has additional codes used. 
|              
└── src: This contains all the codes for TileClipper.         
|   ├── GT: Codes to generate tilelevel ground truth for calibration using Yolov5 and StrongSORT.
|   ├── generateLabelsForAll.py: Generate groundtruth for performance metric using Yolov5
|   ├── runCalibrateOnAllVideos.py: Runs calibration on all videos
|   ├── runTileClipperOnAllVideos.py: Runs TileClipper on all videos
|   ├── calibrate.py
|   ├── capture.sh
|   ├── detect_for_groundtruth.py
|   ├── get_results.ipynb: Generates all plots.
|   ├── main_feed_from_camera.py: Not used now. Need to update.
|   ├── metric_calculator.py: Base code for performance metric calculation
|   ├── script.py: Used to generate tiled videos
|   ├── requirements.txt
|   ├── serverCode.py: Need to fix. Old codebase for server.
|   ├── tileClipper.py: TileClipper source code.
|
└── utils: Has addon scripts and codes.    
```

# Running TileClipper on a video
First of all the video need to be segmented (0.5s at 30fps). And should have 16 tiles (4x4). Use scipt.py for this as `python3 script.py --path-to-mp4 dataset/ --tiles 4x4 --res 1280x720`.

Once we have the tiled segmented videos in `tile_4x4_mp4` folder (script.py creates it). Run TileClipper as:

```
python3 tileClipper.py --tiled-video-dir tiled_4x4_mp4/video_name --percentile-array-filename video_percentile_cluster.pkl --cluster-indices-file video_cluster_index_file.pkl --gamma 1.75
```

The percentile and cluster files are generated using calibration.
Use calibrate.py to get this as:
```
python3 calibrate.py --tiled-video-dir tiled_4x4_mp4/video_name --assets-folder path_to_asset_folder
```
It'll create an `F2s/` folder having the pickle files with the video name. It assumes the tile level ground truths are there in `assets/GroundTruths_TileLevel/` folder. These ground truths can be generated using the files in `src/GT/`. The steps are in a separate README in that folder.

