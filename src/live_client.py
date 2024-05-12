##############################################################################
# This file has the client (camera) handler code which streams videos to the
# server for calibration and inference during the Live Experiment. 
# Raw frames are captured directly from camera using FFmpeg and encoded into 
# tiled videos using Kvazaar and GPAC.
#
# Usage:
#    python3 live_client.py --name nameForTheExperiment 
#                           --runs numberOfSegmentsToStream
##############################################################################

from pathlib import Path
import numpy as np
import subprocess as sp
import time, calibrate, argparse, joblib
import socket
from tileClipper import TileClipper as tc

# Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--tiles", default="4x4", type=str, help="Number of tiles (e.g. 4x4).")
parser.add_argument("--num_of_frames", default=15, type=int, help="Number of frames (at 30fps) to capture from camera (video segment duration default is 15 means 0.5sec).")
parser.add_argument("--video_res", default="1280x720", type=str, help="Video resolution to capture from camera (e.g. 1280x720).")
parser.add_argument("--framerate", default=30, type=int, help="Framerate at which to capture from camera.")
parser.add_argument("--runs", default=100, type=int, help="How many segments to stream?")
parser.add_argument("--name", type=str, default=None, help="Name of the folder to save in.")
opt = parser.parse_args() 
print(opt) 

SERVER_IP = "192.168.226.150"
SERVER_PORT = 12345
SEND_BUFFER_SIZE = 1024
STATIC_TILES = {2}                                                              # Previously decided (by YOLO or other DNN). Tile 2 cannot be removed
TOTAL_TILES = int(opt.tiles.split("x")[0]) * int(opt.tiles.split("x")[1])
DISCONNECT_MESSAGE = b"DISCONNECT"
NUMBER_OF_FRAMES_CAPTURED = opt.num_of_frames
NUMBER_OF_BUFFERED_SEGMENTS = 0
NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION = 60                                                      # Dumped incr array. Only used when require_calibration is True

if opt.name == None:
    raise ValueError("No name specified to --name")

# Function to copy files when not using live setup. Acts as simulator to store locally
def copyFile(file):
    Path(f"../assets/removedTileMp4/{opt.name}").mkdir(exist_ok=True, parents=True) 
    # Path(str(path)[:str(path).find("tiled_4x4_mp4")]+'../assets/removedTileMp4/'+filename).mkdir(parents=True, exist_ok=True)
    sp.run(["cp", f"../assets/tiled_4x4_mp4/{opt.name}/{file}",  f'../assets/removedTileMp4/{opt.name}/{str(Path(file).stem)}'+'_tile_removed_copied.mp4']) # for simulating sending files as it is
    return f'../assets/removedTileMp4/{opt.name}/{str(Path(file).stem)}'+'_tile_removed_copied.mp4'

# Function to save filtered tiles when not using live setup. Acts as simulator to store locally
def removeTiles(file, list_of_tiles_to_remove):
    Path(f"../assets/removedTileMp4/{opt.name}/").mkdir(exist_ok=True, parents=True) 
    lst = [str(list_of_tiles_to_remove[(i//2)-1]) if(i%2==0) else "-rem" for i in range(1,2*len(list_of_tiles_to_remove)+1)]
    sp.run(["MP4Box"] + lst + [f"../assets/tiled_4x4_mp4/{opt.name}/{file}", "-out", f'../assets/removedTileMp4/{opt.name}/{str(Path(file).stem)}'+'_tile_removed.mp4'])
    return f'../assets/removedTileMp4/{opt.name}/{str(Path(file).stem)}'+'_tile_removed.mp4'

# Function to get tile encoded video directly from the attached camera 
def captureFromCamera(out_file_name, camera="/dev/video0", video_size=opt.video_res, framerate=opt.framerate, num_of_frames_to_capture=NUMBER_OF_FRAMES_CAPTURED, tiles=opt.tiles):
    Path(f"../assets/tiled_4x4_mp4/{opt.name}").mkdir(exist_ok=True, parents=True)
    o = sp.run(f"bash capture.sh {framerate} {video_size} {camera} {num_of_frames_to_capture} {tiles} {f'../assets/tiled_4x4_mp4/{opt.name}/' + out_file_name}".split(" "), capture_output=False)

def connectToServer(server_ip, server_port):
    Path("../assets/FeedBack").mkdir(exist_ok=True, parents=True)
    _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _sock.connect((server_ip, server_port))
    print("[i] Connected: Server")
    return _sock

def sendFileToServer(file_name, _socket):
    f = open(file_name, "rb")
    d = f.read(SEND_BUFFER_SIZE)
    while len(d) != 0:
        _socket.send(d)
        d = f.read(SEND_BUFFER_SIZE)
    _socket.send(b"NRL_Sent")
    print("[i] Sent: File")    
    f.close()     

def recvFileFromServer(file_name, _socket):
    # Receives the thresholds as pickled numpy array
    ff = open(file_name, "wb")
    ff.write(_socket.recv(SEND_BUFFER_SIZE))
    print("[i] Received: Thresholds\n")  
    ff.close() 

ii, jj = 0, 0
sock = connectToServer(SERVER_IP, SERVER_PORT)

latency = []
bitrates = np.zeros((NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION, TOTAL_TILES))                                        # To store bitrates during calibration
while True:
    videoSegName = f"output_{str(ii).zfill(6)}.mp4"
    
    ts = time.time()
    # Capturing tiled video segments from camera
    o = captureFromCamera(out_file_name=videoSegName)
    te = time.time()

    ts1 = time.time()                                                       # For calulating runtime
    bitrate = sp.run(["ffprobe", "-v", "error",
                "-show_entries", "stream=bit_rate",
                "-of", "default=noprint_wrappers=1", 
                f"../assets/tiled_4x4_mp4/{opt.name}/{videoSegName}"], stdout=sp.PIPE)
    te1 = time.time()
    arr = np.fromiter(map(lambda x: int(x[9:]), bitrate.stdout.decode().split('\n')[1:-1]), dtype=int) # 9 => 'bit_rate='; [1:-1] => 1 because tile 1 has metadata only
    
    if ii < NUMBER_OF_BUFFERED_SEGMENTS + NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION:
        sendFileToServer(f"../assets/tiled_4x4_mp4/{opt.name}/{videoSegName}", sock)    
        bitrates[jj] = arr
        # Receiving Feedback
        if ii == (NUMBER_OF_BUFFERED_SEGMENTS + NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION) - 1:
            recvFileFromServer(f"../assets/FeedBack/{opt.name}_cluster_indices.pkl", sock)
            recvFileFromServer(f"../assets/FeedBack/f2s_{opt.name}_cluster10.pkl", sock)
            # doneCalibration = True
            clusterIndicesList = tc.readClusterIndicesFileFromServer(f"../assets/FeedBack/{opt.name}_cluster_indices.pkl")
            bestPercentileArray = tc.readBestPercentileFileFromServer(f"../assets/FeedBack/f2s_{opt.name}_cluster10.pkl")
            bestPercentileForClusters = []
            for tile_indx in range(TOTAL_TILES):
                l, u = tc.getBestPercentileForClustersOfATile(bestPercentileArray, tile_indx)
                bestPercentileForClusters.append([l, u]) 
        jj += 1

    else:
        # ts1 = time.time()                                                       # For calulating runtime
        # bitrate = sp.run(["ffprobe", "-v", "error",
        #             "-show_entries", "stream=bit_rate",
        #             "-of", "default=noprint_wrappers=1", 
        #             f"../assets/tiled_4x4_mp4/{opt.name}/{videoSegName}"], stdout=sp.PIPE)
        # te1 = time.time()
        # arr = np.fromiter(map(lambda x: int(x[9:]), bitrate.stdout.decode().split('\n')[1:-1]), dtype=int) # 9 => 'bit_rate='; [1:-1] => 1 because tile 1 has metadata only
        
        if ii == NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION:
            # Reading the percentile file got during calibration and finding best percentile to use for each tile
            _trueCluster, _falseCluster = [], []
            for tile_indx in range(TOTAL_TILES):   
                first_n_true_segments_indx, first_n_false_segments_indx = clusterIndicesList[tile_indx]
                trueCluster, falseCluster = tc.getClustersUsingGaussianForNoObject(first_n_true_segments_indx, first_n_false_segments_indx, bitrates[:NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION, tile_indx])
                _trueCluster.append(trueCluster)
                _falseCluster.append(falseCluster)

        tilesToRemove = []
        for _tile in range(TOTAL_TILES):
            selected, threshold = tc.getSelectedTilesForGaussianScheme(true_cluster=_trueCluster[_tile], false_cluster=_falseCluster[_tile], bitrate=float(arr[_tile]), lower_percentile=bestPercentileForClusters[_tile][0], upper_percentile=bestPercentileForClusters[_tile][1])
            if selected == False:       
                tilesToRemove.append(_tile + 2)   

        # Checking values greater than threshold. And extracting their indices
        tmp = STATIC_TILES.copy()
        tmp.update(tilesToRemove)
        tmp.discard(2)                                                                                          # Tile 2 cannot be removed due to codec constraint
        # te2 = time.time()
        # print(tilesToRemove)

        # Removing unwanted tiles
        if len(tmp) == 0:                                                                                       # no tiles removed, send segment as it is
            copyFile(videoSegName)
        else:
            f = removeTiles(videoSegName, list(tmp))
            # print(data,">>>>>>>>>>>>>>>>>", f)
        te3 = time.time()
    
    ii += 1
    if ii >= opt.runs:                       # Reading from camera
        sock.send(DISCONNECT_MESSAGE)
        print("[i] Sent: Disconnect Message")
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        break
