#######################################################
# Script to crop tiles from a video based on the tile 
# obtained from the tiled videos generated from GPAC.
#######################################################

import utils
from pathlib import Path

# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/DETRAC/Untiled_mp4_30qp/MVI_40192").iterdir()))
# for vid in range(len(p)):
# 	func.cropAllTiles(f"{str(p[vid])}", "/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/DETRAC/tiled_4x4_mp4/MVI_40192/output0000_tiled.mp4", f"cropped/DETRAC/MVI_40192", f"seg_{str(vid).zfill(4)}.mp4")

# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/DETRAC/Untiled_mp4_30qp/MVI_40212").iterdir()))
# for vid in range(len(p)):
# 	func.cropAllTiles(f"{str(p[vid])}", "/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/DETRAC/tiled_4x4_mp4/MVI_40212/output0000_tiled.mp4", f"cropped/DETRAC/MVI_40212", f"seg_{str(vid).zfill(4)}.mp4")


# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/AINormal/Untiled_mp4_30qp/AITr1cam10").iterdir()))
# for vid in range(len(p)):
# 	func.cropAllTiles(f"{str(p[vid])}", "/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/AINormal/tiled_4x4_mp4/AITr1cam10/output0000_tiled.mp4", f"cropped/AINormal/AITr1cam10", f"seg_{str(vid).zfill(4)}.mp4")

# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/AINormal/Untiled_mp4_30qp/AITr1cam16").iterdir()))
# for vid in range(len(p)):
# 	func.cropAllTiles(f"{str(p[vid])}", "/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/AINormal/tiled_4x4_mp4/AITr1cam16/output0000_tiled.mp4", f"cropped/AINormal/AITr1cam16", f"seg_{str(vid).zfill(4)}.mp4")


# Cropping all videos

# # For DETRAC
# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/DETRAC/Untiled_mp4_30qp/").iterdir()))
# for vid in p:
# 	pp = sorted(list(Path(vid).iterdir()))
# 	for seg in range(len(pp)):
# 		func.cropAllTiles(f"{str(pp[seg])}", f"/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/DETRAC/tiled_4x4_mp4/{vid.stem}/output0000_tiled.mp4", f"Videos/cropped/DETRAC/{vid.stem}", f"seg_{str(seg).zfill(4)}.mp4")

# # For AINormal
# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/Videos/AINormal/Untiled_mp4_30qp/").iterdir()))
# for vid in p:
# 	if str(vid) == "AITr1cam10" or str(vid) == "AITr1cam16":
# 		pass
# 	else:
# 		pp = sorted(list(Path(vid).iterdir()))
# 		for seg in range(len(pp)):
# 			func.cropAllTiles(f"{str(pp[seg])}", f"/home/shubhamch/project/EdgeBTS/VISTA/Videos/AINormal/tiled_4x4_mp4/{vid.stem}/output0000_tiled.mp4", f"/home/shubhamch/project/EdgeBTS/VISTA/After INFOCOM/Finiding Optimal/Videos/cropped/AINormal/videos/{vid.stem}", f"seg_{str(seg).zfill(4)}.mp4")

# For AIConditions
p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/Videos/AIConditions/Untiled_mp4_30qp/").iterdir()))
for vid in p:
	pp = sorted(list(Path(vid).iterdir()))
	for seg in range(len(pp)):
		func.cropAllTiles(f"{str(pp[seg])}", f"/home/shubhamch/project/EdgeBTS/VISTA/Videos/AIConditions/tiled_4x4_mp4/{vid.stem}/output0000_tiled.mp4", f"/home/shubhamch/project/EdgeBTS/VISTA/After INFOCOM/Finiding Optimal/Videos/cropped/AIConditions/videos/{vid.stem}", f"seg_{str(seg).zfill(4)}.mp4")

# # For OurRec
# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/OurRec/Untiled_mp4_30qp/").iterdir()))
# for vid in p:
# 	pp = sorted(list(Path(vid).iterdir()))
# 	for seg in range(len(pp)):
# 		func.cropAllTiles(f"{str(pp[seg])}", f"/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/OurRec/tiled_4x4_mp4/{vid.stem}/output0000_tiled.mp4", f"Videos/cropped/OurRec/{vid.stem}", f"seg_{str(seg).zfill(4)}.mp4")

# # For Indian Videos
# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/IndianVideos/Untiled_mp4_30qp/").iterdir()))
# for vid in p:
# 	pp = sorted(list(Path(vid).iterdir()))
# 	for seg in range(len(pp)):
# 		func.cropAllTiles(f"{str(pp[seg])}", f"/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/IndianVideos/tiled_4x4_mp4/{vid.stem}/output0000_tiled.mp4", f"Videos/cropped/IndianVideos/{vid.stem}", f"seg_{str(seg).zfill(4)}.mp4")

# # For YouTube
# p = sorted(list(Path("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/Youtube/Untiled_mp4_30qp/").iterdir()))
# for vid in p:
# 	pp = sorted(list(Path(vid).iterdir()))
# 	for seg in range(len(pp)):
# 		func.cropAllTiles(f"{str(pp[seg])}", f"/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/Youtube/tiled_4x4_mp4/{vid.stem}/output0000_tiled.mp4", f"../Finding Optimal/Videos/cropped/videos/Youtube/{vid.stem}", f"seg_{str(seg).zfill(4)}.mp4")