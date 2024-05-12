########################################################
# Script file remove tiles manually for
# the Static Tile Removal baseline
########################################################
from pathlib import Path
import subprocess as sp
import argparse

# Function to remove tiles using MP4Box
def removeTiles(segFolderName, list_of_tiles_to_remove): # list = [1,2,5,6,8]
    path = Path(segFolderName); Path("StaticallyRemovedTiles").mkdir(parents=True, exist_ok=True)
    Path('StaticallyRemovedTiles/'+str(path.name)).mkdir(parents=True, exist_ok=True)
    for p in sorted(list(path.iterdir())):
        lst = [str(list_of_tiles_to_remove[(i//2)-1]) if(i%2==0) else "-rem" for i in range(1,2*len(list_of_tiles_to_remove)+1)]
        sp.run(["MP4Box"] + lst + [str(p), "-out", 'StaticallyRemovedTiles/'+str(path.name)+'/'+p.stem+"_tile_removed.mp4"])


parser = argparse.ArgumentParser()
parser.add_argument("--input-seg-folder", type=str, help="Input segment folder name")
parser.add_argument("--tiles-to-remove", nargs='+', type=int, help="Tiles to remove")
opt = parser.parse_args() 
print(opt) 

removeTiles(opt.input_seg_folder, opt.tiles_to_remove)