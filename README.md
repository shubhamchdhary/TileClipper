# TileClipper
## Introduction
This repository contains codes/artifacts for the paper "TileClipper: Lightweight Selection of Regions of Interest from Videos forTraffic Surveillance". TileClipper is a system that utilizes tile sampling to substantially reduce bandwidth consumption by selecting spatial regions (tiles) of interest for further analysis at the server. 
![Before Tile Pruning](assets/UnremovedTileFrameSnip.png) | ![After Tile Pruning](assets/tileRemovedFrameSnip1.png)
:--:| :--:
**Before Tile Pruning**| **After Tile Pruning**

## Getting Started

### 1) Directory Structure
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
└── baselines: Has codes for the baselines. 
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

### 2) Dependencies
The codebase uses FFmepg (v4.2.7), GPAC (v2.2.1), and Kvazaar (v2.0.0) for encoding/manipulating a tiled videos. Install FFmpeg using `sudo apt install ffmpeg` command. [Kvazaar](https://github.com/ultravideo/kvazaar) and [GPAC](https://github.com/gpac/gpac/wiki/GPAC-Build-Guide-for-Linux) requires building. Follow the build instructions in their respective repositories. For GPAC, go with a full GPAC build, not the minimal one.

### 3) Creating Python Environment
```bash
$> git clone https://github.com/shubhamchdhary/TileClipper.git
$> git submodule update --init --recursive
$> cd TileClipper
$> pip3 install virtualenv                  
$> python3 -m virtualenv env
$> source env/bin/activate                       # for bash
(env) $> pip3 install -r src/requirements.txt    # installs python libraries
```

### 4) Downloading Dataset
Download the dataset available on [Zenodo](https://doi.org/zenodo/10.5281/zenodo.11179900). Unzip the compressed the file in the current directory. Once done there should be a `videos/` directory having all the necessary pre-processed dataset to reproduce the results.

### 5) Running TileClipper on a sample video
TileClipper operates on tiled videos. The `videos/` folder contains a `TestDataset/` folder with a sample video to validate TileClipper. Run TileClipper as:

```bash
$> python3 src/tileClipper.py --tiled-video-dir videos/TestDataset/tiled_4x4_mp4/AITr1cam10 --percentile-array-filename assets/F2s/f2s_AITr1cam10_cluster10.pkl  --cluster-indices-file assets/F2s/AITr1cam10_cluster_indices.pkl --gamma 1.75
```
Once run, you'll find a `removedTileMp4/` folder inside `videos/TestDataset/` directory. It contains the segmented tiled video with pruned tiles. You can play these using GPAC as `gpac -play video.mp4`. Other video players like VLC cannot decode tiled videos.

Note that the above execution assumes that the calibration is already done to get the percentile and cluster files. To run calibration on a video use `calibrate.py` as:
```bash
$> python3 src/calibrate.py --tiled-video-dir videos/dataset/tiled_4x4_mp4/video_name --assets-folder assets/
```
It'll create an `F2s/` folder inside `assets/` having the pickle files with the video name. It assumes the tile level ground truths are there in `assets/GroundTruths_TileLevel/` folder. These ground truths can be generated using the files in `src/GT/`. The steps are in a separate [README](src/GT/README.md).

### 6) Reproducing results
To quickly reproduce the results, the necessary groundtruths, labels, and processed files are already placed inside the `videos/` and `baselines/` folders.
Utilize the `src/get_results.ipynb` notebook file to generate the plots. Note this notebook file must be run locally not on Google Colab as it parses the dataset to generate results. `get_results.ipynb` file can be run inside VS Code IDE or Jupyter Notebook.

## Experiments

First of all the video need to be segmented (0.5s at 30fps). And should have 16 tiles (4x4). Use script.py for this as `python3 script.py --path-to-mp4 dataset/ --tiles 4x4 --res 1280x720`.

Once we have the tiled segmented videos in `tile_4x4_mp4` folder (script.py creates it).

### Baselines
Follow this [README](baselines/README.md).