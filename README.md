# TileClipper
## Introduction
This repository contains codes/artifacts for the paper "TileClipper: Lightweight Selection of Regions of Interest from Videos forTraffic Surveillance". TileClipper is a system that utilizes tile sampling to substantially reduce bandwidth consumption by selecting spatial regions (tiles) of interest for further analysis at the server. 
![Before Tile Pruning](assets/UnremovedTileFrameSnip.png) | ![After Tile Pruning](assets/tileRemovedFrameSnip1.png)
:--:| :--:
**Before Tile Pruning**| **After Tile Pruning**

## Getting Started

### 1) Directory Structure
```
└── assets 
|   ├── Bitrates                    : Bitrates of all videos
|   ├── F2s                         : Calibrated outputs
|   ├── GroundTruths                : Yolov5x Groundtruths
|   ├── GroundTruths_TileLevel      : StrongSORT-Yolo groundtruths for calibration
|   ├── Runtimes                    : Speed (fps) pickle files 
|   |   ├── On Nano
|   |   ├── On RPi4
|   |
|   ├── labels                      : Labels to calculate accuracy
|   ├── rates.pkl                   
|   ├── ratios.pkl                   
|   ├── UnremovedTileFrameSnip.png               
|   ├── tileRemovedFrameSnip1.png                  
|
└── baselines
|   ├── CloudSeg                    : CloudSeg codes
|   ├── DDS                         : DDS artifacts
|   ├── Reducto                     : Reducto implementation
|   ├── StaticTileRemoval           : STR codes
|   ├── README.md                   
|              
└── src        
|   ├── GT                          : StrongSORT-Yolo codebase
|   ├── calibrate.py                : For TileClipper calibration
|   ├── capture.sh                  : Live tiled video encoding from camera 
|   ├── detect_for_groundtruth.py   : Generate labels/GT using  Yolov5x
|   ├── get_results.ipynb           : Generates all plots.
|   ├── live_client.py              : Camera-side code during live experiment
|   ├── live_server.py              : Server-side code during live experiment
|   ├── metric_calculator.py        : Base code for performance metric calculation
|   ├── tileClipper.py              : TileClipper's source code.
|   ├── ratios_withVideoName.pkl
|   ├── requirements.txt
|
└── utils                           : Has addon scripts and codes.  
|
└── videos                          : Available after downloading and extracting the dataset

```

### 2) Dependencies
All the experiments are designed and tested on Ubuntu 20.04 LTS. Use the same OS to reproduce results. For a differnt Linux distribution, change the commands accordingly. The codebase uses FFmepg (v4.2.7), GPAC (v2.2.1), and Kvazaar (v2.0.0) for encoding/manipulating a tiled videos. Install FFmpeg using `sudo apt install ffmpeg`. [Kvazaar](https://github.com/ultravideo/kvazaar) and [GPAC](https://github.com/gpac/gpac/wiki/GPAC-Build-Guide-for-Linux) requires building. Follow the build instructions in their respective repositories. For GPAC, go with a full GPAC build, not the minimal one. Unless otherwise stated, we use Python 3.8 for all the experiments.

### 3) Creating Python Environment
```bash
$> git clone https://github.com/shubhamchdhary/TileClipper.git
$> cd TileClipper
$> git submodule update --init --recursive
$> pip3 install virtualenv                  
$> python3 -m virtualenv env
$> source env/bin/activate                       # for bash
(env) $> pip3 install -r src/requirements.txt    # installs python libraries
```

### 4) Downloading Dataset
Download the dataset (.zip file approximately 38G) available on [Zenodo](https://doi.org/zenodo/10.5281/zenodo.11179900). Unzip the compressed the file in the current directory. Once done there should be a `videos/` directory having all the necessary pre-processed dataset to reproduce the results. The extracted folder contains the videos from AICC, DETRAC, and OurRec datasets. Due to privacy concerns we have not made the Others and Live Deployment videos public. For evaluation on these videos, we've provided serialized pickle files with video filesizes.

### 5) Running TileClipper on a sample video
TileClipper operates on tiled videos. The `videos/` folder contains a `TestDataset/` folder with a sample video to validate TileClipper. Run TileClipper on it as:

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

### Running Baselines
Follow this [README](baselines/README.md).
