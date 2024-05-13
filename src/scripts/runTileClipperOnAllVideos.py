##############################################
# This script runs TileClipper on videos
##############################################

import subprocess as sp
from pathlib import Path

root = "../../videos/"
datasets = ["AIConditions/tiled_4x4_mp4", "DETRAC/tiled_4x4_mp4", "AINormal/tiled_4x4_mp4", "OurRec/tiled_4x4_mp4"]
gamma = 1.75

# Running TileClipper after calibration
for i in range(len(datasets)):
    for j, data in enumerate(sorted(list(Path(root + datasets[i]).iterdir()))):
        print(f"Running on {data}.")
        sp.run(f"python ../tileClipper.py --tiled-video-dir {data} --percentile-array-filename ../../assets/F2s/f2s_{data.name}_cluster10.pkl --cluster-indices-file ../../assets/F2s/{data.name}_cluster_indices.pkl --gamma {gamma}".split(" "))
        print(f"Done for {data}")



# Running TileClipper on videos without recalibration
path = root + "TileClipper_Without_Recalibration/When_calibrated_at_noon/AIConditions/tiled_4x4_mp4"

sp.run(f"python ../tileClipper.py --tiled-video-dir {path}/cam_1dawn_ --percentile-array-filename ../../assets/F2s/f2s_cam_1_a_se_cluster10.pkl --cluster-indices-file ../../assets/F2s/cam_1_a_se_cluster_indices.pkl --gamma {gamma}".split(" "))
sp.run(f"python ../tileClipper.py --tiled-video-dir {path}/cam_1rain_ --percentile-array-filename ../../assets/F2s/f2s_cam_1_a_se_cluster10.pkl --cluster-indices-file ../../assets/F2s/cam_1_a_se_cluster_indices.pkl --gamma {gamma}".split(" "))

sp.run(f"python ../tileClipper.py --tiled-video-dir {path}/cam_4dawn_ --percentile-array-filename ../../assets/F2s/f2s_cam_4_a_5m_cluster10.pkl --cluster-indices-file ../../assets/F2s/cam_4_a_5m_cluster_indices.pkl --gamma {gamma}".split(" "))
sp.run(f"python ../tileClipper.py --tiled-video-dir {path}/cam_4rain_ --percentile-array-filename ../../assets/F2s/f2s_cam_4_a_5m_cluster10.pkl --cluster-indices-file ../../assets/F2s/cam_4_a_5m_cluster_indices.pkl --gamma {gamma}".split(" "))

sp.run(f"python ../tileClipper.py --tiled-video-dir {path}/cam_5dawn_ --percentile-array-filename ../../assets/F2s/f2s_cam_5_a_5m_cluster10.pkl --cluster-indices-file ../../assets/F2s/cam_5_a_5m_cluster_indices.pkl --gamma {gamma}".split(" "))
sp.run(f"python ../tileClipper.py --tiled-video-dir {path}/cam_5rain_ --percentile-array-filename ../../assets/F2s/f2s_cam_5_a_5m_cluster10.pkl --cluster-indices-file ../../assets/F2s/cam_5_a_5m_cluster_indices.pkl --gamma {gamma}".split(" "))

sp.run(f"python ../tileClipper.py --tiled-video-dir {path}/cam_7dawn_ --percentile-array-filename ../../assets/F2s/f2s_cam_7_a_5m_cluster10.pkl --cluster-indices-file ../../assets/F2s/cam_7_a_5m_cluster_indices.pkl --gamma {gamma}".split(" "))
sp.run(f"python ../tileClipper.py --tiled-video-dir {path}/cam_7rain_ --percentile-array-filename ../../assets/F2s/f2s_cam_7_a_5m_cluster10.pkl --cluster-indices-file ../../assets/F2s/cam_7_a_5m_cluster_indices.pkl --gamma {gamma}".split(" "))

# # Running TileClipper on Live Videos
# videos = "../../videos/live_videos/tiled_4x4_mp4"
# for video in Path(videos).iterdir():
#     print(f"Running on {video.name}.")
#     sp.run(f"python ../tileClipper.py --tiled-video-dir {video} --percentile-array-filename ../../assets/F2s/f2s_{video.name}_cluster10.pkl --cluster-indices-file ../../assets/F2s/{video.name}_cluster_indices.pkl --gamma {gamma}".split(" "))
#     print(f"Done for {video.name}")
