################################################
# This script runs Yolov5 on TileClipper output
# videos on server
################################################

import subprocess as sp
from pathlib import Path

root = "../../videos/aggrMp4/gamma_1.75/"
ignore = ["TileClipper_Without_Recalibration", "day1-live2-post-cal", "day1-live4-10fps"]

# Running Yolov5 to get labels on TileClipper's output videos
for data in Path(root).iterdir():
    if data.name not in ignore:
        print(f"Running on {data.name}.")
        # We've limited the detected COCO classes within 0 to 7 only. These are the object appearing on road. 
        sp.run(f"python3 ../detect_for_groundtruth.py --source {data} --save-txt --save-labelfile-name {data.name}".split(" ") + ["--classes", "0", "1", "2", "3", "4", "5", "6", "7"])
        print(f"Done for {data.name} \n")
    else:
        # pass
        recalib = "TileClipper_Without_Recalibration/When_calibrated_at_noon/"
        for video in Path(root + recalib).iterdir(): 
            print(f"Running on {video.name}_recalib.")
            sp.run(f"python3 detect_for_groundtruth.py --source {video} --save-txt --save-labelfile-name {video.name}_recalib".split(" ") + ["--classes", "0", "1", "2", "3", "4", "5", "6", "7"])
            print(f"Done for {video.name}_recalib \n")        