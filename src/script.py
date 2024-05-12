#############################################################
# Scipt to generate tiled videos.
# Place all mp4s encoded at 30fps in a mp4/ folder.
# Then run this script as "python3 script.py". Note that 
# it requires lots of space (GBs) because it extracts raw 
# videos.
# E.g. if folder structure is dataset/mp4/video.mp4, then run
#	python3 script.py --path-to-mp4 dataset/
#					  --res 1280x720
#					  --tiles 4x4
# Tiled videos will be generated inside tiled_4x4_mp4 folder.
# The rawY4M/, segmented_0.5/, and hevcFromy4m/ folder can be
# deleted as they are redundant. 
#############################################################

from pathlib import Path
import subprocess as sp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path-to-mp4", type=str, help="Path to mp4/ folder of the dataset")
parser.add_argument("--res", type=str, help="Video Resolution, e.g. 1280x720")
parser.add_argument("--tiles", default="4x4", type=str, help="Number of tiles to keep, e.g., 4x4")
opt = parser.parse_args()

path = Path(opt.path_to_mp4)
res = opt.res
tiles = opt.tiles

################################################ Raw extraction ###################################################
for d in sorted(list(Path(path/"mp4").iterdir())):
	print("Extracting Raw......")
	Path(path/"rawY4M").mkdir(parents=True, exist_ok=True)
	p = Path(path/"rawY4M")
	sp.run(["ffmpeg", "-i", str(d), "-pix_fmt", "yuv420p", str(p)+"/"+str(d.stem)+".y4m"])
####################################################################################################################


################################################# Segmenting Raw ###################################################
for d in sorted(list(Path(path/"rawY4M").iterdir())):
	print("Segmenting.....")
	Path(path/("segmented_0.5")).mkdir(parents=True, exist_ok=True)
	Path(path/("segmented_0.5/"+str(d.stem)+"_segmented")).mkdir(parents=True, exist_ok=True)
	p = Path(path/("segmented_0.5/"+str(d.stem)+"_segmented"))
	sp.run(["ffmpeg", "-i", str(d), "-f", "segment", "-segment_time", "0.5", "-pix_fmt", "yuv420p", str(p)+"/"+"output%4d.y4m"])
####################################################################################################################


############################################# Hevc and tiled mp4 conversion ########################################
# For generating tiled Mp4 ############################################
for indx, d in enumerate(sorted(list(Path(path/"segmented_0.5").iterdir()))):
	Path(path/"hevcFromy4m").mkdir(parents=True, exist_ok=True)
	Path(path/("hevcFromy4m/"+str(d.stem)[0:10])).mkdir(parents=True, exist_ok=True)
	p = Path(path/("hevcFromy4m/"+str(d.stem)[0:10]))

	Path(path/"tiled_4x4_mp4").mkdir(parents=True, exist_ok=True)
	Path(path/("tiled_4x4_mp4/"+str(d.stem)[:10])).mkdir(parents=True, exist_ok=True)
	pp = Path(path/("tiled_4x4_mp4/"+str(d.stem)[:10]))

	for dd in Path(str(d)).iterdir():
		print("Converting to HEVC....") 
		sp.run(["kvazaar", "-i", str(dd), "--input-res", res, "--input-fps", "30", "--qp", "30", "--tiles", tiles, "--slice", "tiles", "--mv-constraint", "frametilemargin", "-o", str(p)+"/"+str(dd.stem)+".hevc"])
		print("Converting to tiled mp4")
		sp.run(["MP4Box", "-add", str(p)+"/"+str(dd.stem)+".hevc"+":"+"split_tiles", '-new', str(pp)+"/"+str(dd.stem)+"_tiled.mp4"])
	

# For generating Untiled Mp4 ###########################################
# for indx, d in enumerate(sorted(list(Path(path/"segmented_0.5").iterdir()))):
# 	Path(path/"Untiled_hevc_22qp").mkdir(parents=True, exist_ok=True)
# 	Path(path/("Untiled_hevc_22qp/"+str(d.stem)[0:10])).mkdir(parents=True, exist_ok=True)
# 	p = Path(path/("Untiled_hevc_22qp/"+str(d.stem)[0:10]))

# 	Path(path/"Untiled_mp4_22qp").mkdir(parents=True, exist_ok=True)
# 	Path(path/("Untiled_mp4_22qp/"+str(d.stem)[:10])).mkdir(parents=True, exist_ok=True)
# 	pp = Path(path/("Untiled_mp4_22qp/"+str(d.stem)[:10]))
	
# 	for dd in Path(str(d)).iterdir():
# 		print("Converting to HEVC....") 
# 		# sp.run(["kvazaar", "-i", str(dd), "--input-res", res, "--input-fps", "30", "--qp", "30", "-o", str(p)+"/"+str(dd.stem)+".hevc"]) 
# 		sp.run(["kvazaar", "-i", str(dd), "--input-res", res, "--input-fps", "30", "-o", str(p)+"/"+str(dd.stem)+".hevc"]) 
# 		print("Converting to tiled mp4")
# 		sp.run(["MP4Box", "-add", str(p)+"/"+str(dd.stem)+".hevc", '-new', str(pp)+"/"+str(dd.stem)+"_untiled.mp4"])
####################################################################################################################
