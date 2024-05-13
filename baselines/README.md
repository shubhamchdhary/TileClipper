# Running the baslines
## StaticTileRemoval (STR)
To run STR on a video use: 
``` bash
(env) $> python3 staticTileRemoval.py --input-seg-folder videos/dataset/tiled_4x4_mp4/video_name --tiles-to-remove 3 4 5 6
```
Note that the provided tiles to remove should be from 3 to 17. Tile indexing in GPAC starts with 2, so 16 tiles will have indices from 2 to 17. Tile 2 should not be removed as it contains the necessary offset for decoding other tiles.

To run STR on all video use script as:
``` bash
(env) $> python3 removeTiles.py
```
It will create a `StaticallyRemovedTiles/` directory with all the videos having pruned tiles.

## CloudSeg
Use `python3 cloudSeg_driver.py --input_hr_videodataset_path videos/dataset/ --scale 2 --cuda` to run CloudSeg on a dataset. It will create two folders `sr_videos_x2` and `lr_videos_x2` having super-resolution and low-resolution videos respectively.

Execute `run_cloudSeg.sh` to let CloudSeg run on all dataset videos.

## Reducto
Use `python3 reducto.py` to run Reducto on a sample video. It will create a folder having the video name which contains the relevant frames inside `frames/` folder. Run `python3 encoder.py` to generate videos from these frames. Change the video name inside the `driver()` function in the `reducto.py` and `encoder.py` to run on different videos. We've already run it on all of the videos of all our datasets and provided a [Sheet](Reducto/Reducto-results.xlsx) with the results. 

## DDS
For the first phase of DDS we just the the `detect_for_groundtruth.py`. We send the objects having low confidence score, in the second phase We've provided the output results of DDS in different sheets inside the `DDS/` directory. 