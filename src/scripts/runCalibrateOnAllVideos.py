#####################################################
# This script runs TileClipper calibration on videos
#####################################################

import subprocess as sp
from pathlib import Path

root = "../../videos/"
videos = ["AIConditions/tiled_4x4_mp4", "DETRAC/tiled_4x4_mp4", "AINormal/tiled_4x4_mp4", "OurRec/tiled_4x4_mp4"]

# # toExclude = ["AITr1cam13", "AITr1cam6s", "AITr5S1C3_", "MVI_40211", "MVI_40213", "cam_1dawn_", "cam_1rain_", "joinedS1_s", "joinedS2_s", "joinedS3_s"]

# Running TileClipper calibration
for i in range(len(videos)):
    for j, data in enumerate(sorted(list(Path(root + videos[i]).iterdir()))):
        # if data.name in toExclude:
        print(f"Running calibratio on {data}.")
        sp.run(f"python ../calibrate.py --tiled-video-dir {data} --assets-folder ../../assets".split(" "))
        print(f"Done for {data}")
        # else:
            # print(f"Ignoring: {data.name}")


# # Calibration for live videos
# videos = ".../../videos/live_videos/tiled_4x4_mp4"
# for video in Path(videos).iterdir():
#     print(f"Running calibration on {video.name}.")
#     sp.run(f"python ../calibrate.py --tiled-video-dir {video} --assets-folder ../../assets".split(" "))
#     print(f"Done for {video.name}")