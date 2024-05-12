from pathlib import Path
# import os
import subprocess as sp

def encode(path_for_frames, path_for_vid_segs):
    Path(path_for_vid_segs).mkdir(parents=True, exist_ok=True)
    P = Path(path_for_frames).iterdir()
    for cnt, direc in enumerate(P):
        # print(str(direc))
        path_for_vid = f'{str(direc)}' + '/frame_%05d.bmp'
        command = f'ffmpeg -y -framerate 30 -i {path_for_vid} -c:v libx264 -qp 30 {path_for_vid_segs}/segment_{str(cnt).zfill(5)}.mp4'
        # os.system(command)
        sp.run(command.split(" "))
        #print(cnt, command)
        # cnt += 1


#call encode(path_to_frames, path_to_store_segments)
encode("AITr1cam6n/frames", "AITr1cam6n/outputVideo")
