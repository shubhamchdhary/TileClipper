## How to use StrongSORT-YOLO to generate tile level GroundTruth for TileClipper's calibration? 
Assuming you have a freshly created python (>= Python3.8 tested with v3.8.18) environment. Follow the steps below. NOTE: This env should be different from the one used for TileClipper. Use virtualenv module in Python to create virtual environments as `python -m virtualenv env`.
```bash
(env)$> cd src/GT/
(env)src/GT$> git submodule update --init --recursive
(env)src/GT$/> cp detectTiles_StrongSORT.py IoUBasedMotionTracker_StrongSORT.py StrongSORTYOLO/
(env)src/GT$> cd StrongSORTYOLO
(env)src/GT/StrongSORTYOLO$> pip install -r requirements.txt
```

After this, unzip the `ffmpeg_lib.zip` file and copy the two folders in the unzipped folder into your `env/lib/python3x.x/site-packages/` path. This requires FFmpeg (v4.2.7) to run. Therefore, install it if you don't have using 
```bash
$> sudo apt install ffmpeg=4.2.7
```

If things go well. You'll have Yolov5 based StrongSORT code running. To generate the ground truth use
```bash
(env) src/GT/StrongSORTYOLO$> python detectTiles_StrongSORT.py  --source videoSegmentFolderName/ --save-txt --tiled-video ATiledSegmentOfSameVideo.mp4 --classes 0 1 2 3 4 5 6 7 --save-labelfolder-name folderNameWhereToSaveOutPut/ --yolo-weight weights/yolov5x.pt
```
*Note: The source should always be a folder containing segmented non-tiled videos.*

Running on an example video
```bash
(env) src/GT/StrongSORTYOLO$> python detectTiles_StrongSORT.py  --source ../../../videos/DETRAC/Untiled_mp4_30qp/MVI_39761 --save-txt --tiled-video ../../../videos/DETRAC/tiled_4x4_mp4/MVI_39761/output0000_tiled.mp4 --classes 0 1 2 3 4 5 6 7 --save-labelfolder-name ./ --yolo-weight weights/yolov5x.pt
```
It requires GPU (with CUDA) for faster inference. We already run StrongSORT-YOLO on all the videos. The outputs are provided in the `assets/GroundTruths_TileLevel` folder.

