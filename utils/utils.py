###############################################################
# This file has functions to crop tiles from a video.
# To crop it getd the tile dimensions from tiled videos of
# GPAC.
# And, contains different functions to operate on tiled videos.
###############################################################

import ffmpeg as ff
import subprocess as sp
from pathlib import Path

def getCropDimsForTiles(video):
    '''
    For 16 tiles returns a dict with dimension of each tile
    video: tiled video 
    return: dict with tile dimensions
    '''
    vid = ff.probe(video)
    _detect = {}
    f, w, h = 1, 0, 0
    for i in range(1, 16+1):              # because 17 tiles excluding tile 1
        # f += 1
        ww = vid['streams'][i]['width']
        hh = vid['streams'][i]['height']
        _detect.update({i+1: [ww, hh, w, h]}) # +1 to match tile indexing in GPAC
        w += ww
        if i%4 == 0: 
            h = h+hh
        if i%4 == 0: 
            w = 0 
    return _detect
# print(getCropDimsForTiles("/home/shubhamch/project/EdgeBTS/VISTA/After ICDCS/Videos/DETRAC/tiled_4x4_mp4/MVI_40192/output0000_tiled.mp4"))


def crop(video, crop_box, output_video_name):
    '''
    video: untiled video to crop
    crop_box: crop dimension in an array like [w,h,x,y]
    output_video_name: output video name with path, e.g., seg1/tile2.mp4
    '''
    sp.run(["ffmpeg", "-hide_banner", "-loglevel", "quiet", "-i", video, "-filter:v", f"crop={crop_box[0]}:{crop_box[1]}:{crop_box[2]}:{crop_box[3]}", output_video_name])


def cropAllTiles(video, tiled_video, output_path, out_vid_name):
    '''
    video: untiled video to crop
    tiled_video: tiled video to get tiles dimension
    output_path: path where to save the video, e.g., seg1/tile2/
    out_vid_name: output video name
    '''
    boxes = getCropDimsForTiles(tiled_video)      # gets co-ordinates of all tiles
    for i in range(2, 18):
        Path(output_path+f"/tile_{str(i).zfill(2)}").mkdir(exist_ok=True, parents=True)
        crop(video, boxes[i], output_path+f"/tile_{str(i).zfill(2)}/"+out_vid_name)

# Function to remove tiles using MP4Box
def removeTiles(file, list_of_tiles_to_remove): # list = [1,2,5,6,8]
    # Need to have a removedTileMp4 folder where tiled_mp4_4x4 is present
    path = Path(file)
    lst = [str(list_of_tiles_to_remove[(i//2)-1]) if(i%2==0) else "-rem" for i in range(1,2*len(list_of_tiles_to_remove)+1)]
    Path('removedTileMp4').mkdir(parents=True, exist_ok=True)
    sp.run(["MP4Box"] + lst + [str(path), "-out", 'removedTileMp4/'+path.stem+"_tile_removed.mp4"])
    return 'removedTileMp4/'+path.stem+"_tile_removed.mp4"

def aggrTile(file, folderNameToSaveIn): # Aggregate tiled mp4 tiles
	path = Path(file); Path('aggrMp4').mkdir(parents=True, exist_ok=True)
	Path('aggrMp4/'+folderNameToSaveIn).mkdir(parents=True, exist_ok=True)
	sp.run(["gpac","-i", str(path),"tileagg", "@", "-o", f'aggrMp4/{folderNameToSaveIn}/'+path.stem+"_tileagg.mp4"]);return 'aggrMp4/'+path.stem+"_tileagg.mp4"

def reencode(file):
    path = Path(file)
    Path('reencodedMP4').mkdir(parents=True, exist_ok=True)
    sp.run(["ffmpeg", "-i", str(path), '-v', 'error', "-c:v", "libx265", '-x265-params', 'log-level=error', 'reencodedMP4/'+path.stem+"_ffmpeg.mp4"])
    return 'reencodedMP4'+path.stem+"_ffmpeg.mp4"

def removeFile(file): # Deletes a file
	Path(file).unlink()

# conatenation using ffmpeg
def concatenateFiles(filePath, saveName="final.mp4"):
	Path('concatenatedMp4').mkdir(parents=True, exist_ok=True)
	# sp.run(["rm", "-rf", "aggrMp4"])
	if Path("out.txt").exists() == True:
		removeFile("out.txt")
		# print("Not Found")
	with open("out.txt","a") as f:
		for i in sorted((list(Path(filePath).iterdir()))):
			f.write("file "+str(filePath)+"/"+i.stem+".mp4"+"\n")
	sp.run(["ffmpeg", "-v", "0",
				"-f", "concat",
				"-i", "out.txt", 
				"-c:v", "libx265", '-x265-params', 'log-level=error', 'concatenatedMp4/'+saveName])
	removeFile("out.txt")
	sp.run(["rm", "-rf", "aggrMp4"])

# concatenation using MP4Box
def concatenateFilesMP4Box(filePath, saveName="final.mp4"):
	Path('concatenatedMp4').mkdir(parents=True, exist_ok=True)
	p = sorted(list(Path(filePath).iterdir()))
	l = []
	for i in p:
		l.append("-cat")
		l.append(str(i))
	# print(['MP4Box'] + l + ['concatenatedMp4/'+saveName])
	sp.run(['MP4Box'] + l + ['concatenatedMp4/'+saveName])
	# sp.run(["rm", "-rf", "aggrMp4"])

def concatAllVideos(path):
    pp = sorted(list(Path(path).iterdir()))
    for j in pp: 
        p = sorted(list(Path(str(j)).iterdir()))
        for i in p:
            aggrTile(str(i))
        # concatenateFiles("aggrMp4", str(j.stem)+"_final.mp4")
        concatenateFilesMP4Box("aggrMp4", str(j.stem)+"_final.mp4")

def concatSingleVideo(path):
    p = sorted(list(Path(path).iterdir()))
    for i in p:
        aggrTile(str(i))
    # concatenateFiles("aggrMp4", str(j.stem)+"_final.mp4")
    concatenateFilesMP4Box("aggrMp4", Path(path).stem+"_final.mp4")

def aggregateSingleVideoSegments(path, folderNameToSaveIn):
    p = sorted(list(Path(path).iterdir()))
    for i in p:
        aggrTile(str(i), folderNameToSaveIn)

def removeTilesOfSingleVideoSegments(path, tilesList): # tileList = [3,4,6,7] should not include 1 and 2
    p = sorted(list(Path(path).iterdir()))
    for i in p:
        removeTiles(str(i), tilesList)