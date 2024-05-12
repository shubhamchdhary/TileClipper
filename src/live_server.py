####################################################################
# This file has the server handler code for the camera side (client)
# which streams videos for calibration and inference during the Live 
# Experiment. 
#
# Usage:
#    python3 live_server.py --name nameToKeepForTheExperiment
####################################################################

import socket, argparse
from pathlib import Path
import subprocess as sp
from tqdm import tqdm

# Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name of the folder to save in.")
opt = parser.parse_args() 
print(opt) 

SERVER_IP = "127.0.0.1"
SERVER_PORT = 1234
SEND_BUFFER_SIZE = 1024
DISCONNECT_MESSAGE = b"DISCONNECT"
NUMBER_OF_BUFFERED_SEGMENTS = 0
NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION = 60

# Setting up socket connection
socketConection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socketConection.bind((SERVER_IP, SERVER_PORT))
socketConection.listen(2)

connection, address = socketConection.accept()
print(f"[i] Connected: {address}")

def convertToUntiledMp4(tiledVideo: str, untiledVideoName: str) -> str:
	# Function to covert tiled video into untiled mp4
	path = Path(tiledVideo)
	Path(f'aggrMp4Server/{str(untiledVideoName)}').mkdir(parents=True, exist_ok=True)
	sp.run(["gpac","-i", tiledVideo, "tileagg", "@", "-o", 'aggrMp4Server/'+str(untiledVideoName)+"/"+str(path.stem)+"_untiled.mp4"])
	return 'aggrMp4Server/'+str(untiledVideoName)+"/"+str(path.stem)+"_untiled.mp4"


def sendFileToClient(file_name: str, _socket: socket) -> None:
	# Sends the thresholds as pickled numpy array
	f = open(file_name, "rb")
	_socket.send(f.read(SEND_BUFFER_SIZE)) 
	print("[i] Sent: File")   
	f.close()     

def recvFileFromClient(file_name: str, _socket: socket) -> bool:
	close = False
	ff = open(file_name, "wb")
	dd = _socket.recv(SEND_BUFFER_SIZE)
	while True:
		ff.write(dd)
		dd = _socket.recv(SEND_BUFFER_SIZE)
		# print(len(dd), ">>>>>>>>>>>>>>>>")
		if len(dd) < SEND_BUFFER_SIZE and dd[-8:] == b"NRL_Sent":							# String "NRL_Sent" acts as EOF for the received stream
			ff.write(dd[:-8])
			# print("[i] Received: File")  
			break
		elif len(dd) < SEND_BUFFER_SIZE and dd[-10:] == DISCONNECT_MESSAGE:				# String "DISCONNECT" to detect close call from client
			ff.write(dd[:-10])
			# print("[i] Received: Disconnect Call")  
			_socket.shutdown(socket.SHUT_RDWR)
			_socket.close()
			close = True
			break
	ff.close() 
	return close
    
ii = 0
Path(f"receivedVideos/{opt.name}").mkdir(exist_ok=True, parents=True) 
Path(f"output/GroundTruths_TileLevel").mkdir(exist_ok=True, parents=True) 
calPBar = tqdm(total=NUMBER_OF_BUFFERED_SEGMENTS + NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION)
while True:
	videoName = f"receivedVideos/{opt.name}/output_{str(ii).zfill(6)}.mp4"
	close = recvFileFromClient(videoName, connection)
	convertToUntiledMp4(videoName, opt.name)			# converts tiled video to a normal video becuase Yolov5 cannot decode tiled videos
	# print(f"[i] Received: File {ii}")
	calPBar.update()  
     
	if close:
		print("[i] Received: Disconnect Call")  
		calPBar.close()
		break

	if ii == (NUMBER_OF_BUFFERED_SEGMENTS + NUMBER_OF_BUFFERED_SEGMENTS_DURING_CALIBRATION) - 1:
		calPBar.close()
		# Generating Tilelevel GroundTruth 
		print("\nGenerating Groundtruths >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
		sp.run(f"python3 GT/StrongSORTYOLO/detectTiles_StrongSORT.py  --source aggrMp4Server/{opt.name} --save-txt --tiled-video receivedVideos/{opt.name}/output_000000.mp4 --classes 0 1 2 3 4 5 6 7 --save-labelfolder-name output/GroundTruths_TileLevel/ --yolo-weight weights/yolov5x.pt".split(" "))

		# Running Calibrations
		print("\nRunning: Calibration >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
		sp.run(f"python3 calibrate.py --tiled-video-dir receivedVideos/{opt.name} --assets-folder output".split(" "))
		sendFileToClient(f"output/F2s/{opt.name}_cluster_indices.pkl", connection)
		sendFileToClient(f"output/F2s/f2s_{opt.name}_cluster10.pkl", connection)
	ii += 1
