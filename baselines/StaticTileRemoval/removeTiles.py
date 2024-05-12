# This script removes tiles statically from all videos

import subprocess as sp
from pathlib import Path

videosWithRemovalTiles = {"MVI_39761" : "3 5", "MVI_40211" : "3", "MVI_40212" : "3", "MVI_40213" : "3 4",
                          "MVI_40732" : "3 4 5", "MVI_40991" : "4 5 9 10 11 14 15 16", "MVI_40992" : "4 5 9 10 11 14 15 16", "MVI_41063" : "4 5",
                          "MVI_63554" : "4 5",
                          "AITr1cam8_" : "14", "AITr1cam13" : "3 4 5", "AITr1cam14_2560" : "3 4 5 6 7 8 9", "AITr1cam15" : "4 5 10 14 15 16",
                          "AITr1cam16" : "5 9", "AITr1cam17" : "5 6 10 14", "AITr5S1C1_" : "3 5", "AITr5S1C2_" : "5",
                          "AITr5S1C3_" : "5", "AITr5S1C4_" : "5", "AITr5S3C10" : "5 9", "AITr5S3C14" : "5 6", "AITr5S3C15" : "4 5 8 9",
                          "AITr1cam6n" : "4 5", "AITr1cam6s" : "4 5", "cam_1_a_se" : "3 14 15 16 17", "cam_1dawn_" : "3 14 15 16 17",
                          "cam_1rain_" : "3 14 15 16 17", "cam_7_a_5m" : "3 6 17", "cam_7dawn_" : "3 6 17", "cam_7rain_" : "3 6 17",
                          "joinedS1_s" : "5 9", "joinedS1_s" : "5 9"}

datasets = ["AIConditions", "AINormal", "DETRAC", "OurRec"]
for d in datasets:
    path = Path(f"../../videos/{d}/tiled_4x4_mp4").iterdir()
    for video in path:
        if (video.stem) in videosWithRemovalTiles.keys():
            sp.run(f"python staticTileRemoval.py --input-seg-folder {str(video)} --tiles-to-remove {videosWithRemovalTiles[str(video.stem)]}".split(" "))