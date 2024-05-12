# Running the baslines
## StaticTileRemoval (STR)
To run STR on a video use: 
``` bash
$> python staticTileRemoval.py --input-seg-folder videos/dataset/tiled_4x4_mp4/video_name --tiles-to-remove 3 4 5 6
```
Note that the provided tiles to remove should be from 3 to 17. Tile indexing in GPAC starts with 2, so 16 tiles will have indices from 2 to 17. Tile 2 should not be removed as it contains the necessary offset for decoding other tiles.

To run STR on all video use script as:
``` bash
$> python3 removeTiles.py
```
It will create a `StaticallyRemovedTiles/` directory with all the videos having pruned tiles.

## CloudSeg
Use `python3 cloudSeg_driver.py --input_hr_videodataset_path videos/dataset/ --scale 2 --cuda` to run CloudSeg on a dataset. It will create two folders `sr_videos_x2` and `lr_videos_x2` having super-resolution and low-resolution videos respectively.

Execute `run_cloudSeg.sh` to let CloudSeg run on all dataset videos.

## Reducto


## DDS