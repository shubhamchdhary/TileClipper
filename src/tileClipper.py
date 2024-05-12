#################################################################################################
# This scipt has main source code for TileClipper which uses cluster based strategy to filter 
# unwanted tiles. It requires GPAC's MP4Box for tile manipulation. 
# Assuming that the calibration phase is already done separately, run TileClipper as below:
#   python tileClipper.py --tiled-video-dir dataset/tiled_4x4_mp4/videoName 
#                         --percentile-array-filename ../assets/F2s/f2s_videoName__cluster10.pkl 
#                         --cluster-indices-file ../assets/F2s/videoName_cluster_indices.pkl
#################################################################################################

from __future__ import annotations
from pathlib import Path
import numpy as np
import subprocess as sp
import time, argparse
from tqdm import tqdm
import joblib as jb
from calibrate import EvictingQueue


class TileClipper():
    """
    Implements TileClipper that removes tiles without any objects
    in a tiled video based on the statistics of past few segments' bitrates. 
    """
    def __init__(self, static_tiles: set = {2}, total_tiles: int = 16, number_of_calibration_segments: int = 60, object_ratio_limit: float = 0.1, cluster_size: int = 10, gamma: float = 1.75):
        self.staticTiles = static_tiles                                                                         # Tiles that'll always be removed. Tile 2 cannot be removed because of codec constraint.
        self.totalTiles = total_tiles
        self.numberOfCalibrationSegments = number_of_calibration_segments
        self.clusterSize = cluster_size
        self.gamma = gamma
        self.objectRatioLimit = object_ratio_limit


    def removeTiles(self, video_name: str, filename: str, list_of_tiles_to_remove: list) -> str:                # e.g. list = [1,2,5,6,8]
        """
        Function to remove tiles using MP4Box. Assumed the directory
        structure of the tiled video to be tiled_4x4_mp4/videoName/segments.mp4.
        Save the removed tile video in removedTileMp4/videoName/segments.mp4 in the
        same directory as tiled_4x4_mp4/.
        """
        path = Path(filename)
        lst = [str(list_of_tiles_to_remove[(i//2)-1]) if(i%2==0) else "-rem" for i in range(1,2*len(list_of_tiles_to_remove)+1)]; Path(str(path)[:str(path).find("tiled_4x4_mp4")]+'removedTileMp4/'+video_name).mkdir(parents=True, exist_ok=True)
        sp.run(["MP4Box"] + lst + [str(path), "-quiet", "-out", str(path)[:str(path).find("tiled_4x4_mp4")]+'removedTileMp4/'+video_name+'/'+path.stem+"_tile_removed.mp4"], stdout = sp.DEVNULL, stderr = sp.DEVNULL)
        return str(path)[:str(path).find("tiled_4x4_mp4")]+'removedTileMp4/'+video_name+'/'+path.stem+"_tile_removed.mp4"


    def copyFile(self, video_name, filename: str) -> None:
        """
        Function to copy a video segment if no tile can be removed.
        Copied to removedTileMp4/ folder.
        """
        path = Path(filename)
        Path(str(path)[:str(path).find("tiled_4x4_mp4")]+'removedTileMp4/'+video_name).mkdir(parents=True, exist_ok=True)
        sp.run(["cp", filename, str(path)[:str(path).find("tiled_4x4_mp4")]+'removedTileMp4/'+video_name], stdout = sp.DEVNULL, stderr = sp.DEVNULL) # for simulating sending files as it is


    def readBestPercentileFileFromServer(self, file_name: str) -> np.ndarray:
        """
        Reads the best percentile file generated during calibration.
        """
        return np.array(jb.load(file_name))


    def readClusterIndicesFileFromServer(self, file_name: str) -> tuple:
        """
        Reads the file from server to get the indices of bitrates of 
        both the clusters got during calibration.
        """
        return jb.load(file_name)


    def getBestPercentileForClustersOfATile(self, percentile_array: np.ndarray, tile_num: int) -> list:
        """
        Returns the best cluster percentile to use for a tile using the 
        values got during calibration.
        """
        tmp = np.where(percentile_array[:, 2] == tile_num)[0]
        return percentile_array[tmp][np.argmax(percentile_array[tmp, 3])][:2]


    def getSelectedTilesForGaussianScheme(self, true_cluster: list, false_cluster: list, bitrate: float, lower_percentile: int, upper_percentile: int) -> tuple[bool, int]:
        """
        Selects tile using cluster based classification with
        outlier detection (gamma*sigma) for tiles with obj_ratio < 0.1
        """
        if false_cluster != None:
            currentThreshold = (np.percentile(list(true_cluster), lower_percentile) + np.percentile(list(false_cluster), upper_percentile)) / 2
        else:
            currentThreshold = np.median(list(true_cluster)) + (self.gamma * np.std(list(true_cluster)))
            true_cluster.append(bitrate)

        selected = False
        if bitrate > currentThreshold:
            selected = True
            if false_cluster != None:
                true_cluster.append(bitrate)
        else:
            if false_cluster != None:
                false_cluster.append(bitrate)
        return selected, currentThreshold
    

    def getClustersUsingGaussianForNoObject(self, first_n_true_cluster_indxs: np.ndarray, first_n_false_cluster_indxs: np.ndarray, bitrates_during_calibration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns true and false clusters. Each cluster is a single ended queue.
        """
        if (len(first_n_true_cluster_indxs) / self.numberOfCalibrationSegments) < self.objectRatioLimit:
            _trueCluster = EvictingQueue(list(bitrates_during_calibration), size = self.numberOfCalibrationSegments)
            _falseCluster = None
        else:
            _trueCluster = EvictingQueue(list(bitrates_during_calibration[first_n_true_cluster_indxs]), size = self.clusterSize)
            if len(first_n_false_cluster_indxs) != 0:
                _falseCluster = EvictingQueue(list(bitrates_during_calibration[first_n_false_cluster_indxs]), size = self.clusterSize)
            else:
                _falseCluster = EvictingQueue([min(bitrates_during_calibration[first_n_true_cluster_indxs])], size = self.clusterSize)
   
        return _trueCluster, _falseCluster
    

    def run(self, tiled_video_segment_folder: str, percentile_array_file_from_server: str, cluster_indices_file_from_server: str, save_runtime: bool = False) -> None:
        """
        Runs TileClipper on the tiled_video_segment_folder/.
        Uses the best percentile found using the file received
        from server (percentile_array_file_from_server and 
        cluster_indices_file_from_server)
        """
        tileVideoName = Path(tiled_video_segment_folder)
        tiledSegments = sorted(list(tileVideoName.iterdir()))
        runtimeArr = []

        clusterIndicesList = self.readClusterIndicesFileFromServer(cluster_indices_file_from_server)
        bestPercentileArray = self.readBestPercentileFileFromServer(percentile_array_file_from_server)
        bestPercentileForClusters = []
        for tile_indx in range(self.totalTiles):
            l, u = self.getBestPercentileForClustersOfATile(bestPercentileArray, tile_indx)
            bestPercentileForClusters.append([l, u]) 

        bitrates = np.zeros((self.numberOfCalibrationSegments, self.totalTiles))                                        # To store bitrates during calibration

        for i, data in enumerate(tqdm(tiledSegments)):
            ts1 = time.time()                                                                                           # For calulating runtime
            bitrate = sp.run(["ffprobe", "-v", "error",
                        "-show_entries", "stream=bit_rate",
                        "-of", "default=noprint_wrappers=1", 
                        data], stdout=sp.PIPE)
            # te1 = time.time()
            # print(bitrate.stdout.decode().split('\n'))

            arr = np.fromiter(map(lambda x: int(x[9:]), bitrate.stdout.decode().split('\n')[1:-1]), dtype=int)          # 9 => 'bit_rate='; [1:-1] => 1 because tile 1 has metadata only

            if i < self.numberOfCalibrationSegments:
                bitrates[i] = arr
            else:
                if i == self.numberOfCalibrationSegments:
                    # Reading the percentile file got during calibration and finding best percentile to use for each tile
                    _trueCluster, _falseCluster = [], []
                    for tile_indx in range(self.totalTiles):   
                        first_n_true_segments_indx, first_n_false_segments_indx = clusterIndicesList[tile_indx]
                        trueCluster, falseCluster = self.getClustersUsingGaussianForNoObject(first_n_true_segments_indx, first_n_false_segments_indx, bitrates[:self.numberOfCalibrationSegments, tile_indx])
                        _trueCluster.append(trueCluster)
                        _falseCluster.append(falseCluster)

                # ts2 = time.time() 
                tilesToRemove = []
                for _tile in range(self.totalTiles):
                    selected, threshold = self.getSelectedTilesForGaussianScheme(true_cluster=_trueCluster[_tile], false_cluster=_falseCluster[_tile], bitrate=float(arr[_tile]), lower_percentile=bestPercentileForClusters[_tile][0], upper_percentile=bestPercentileForClusters[_tile][1])
                    if selected == False:       
                        tilesToRemove.append(_tile + 2)                                                                 # +2 because GPAC tile indexing starts with 2

                # Checking values greater than threshold. And extracting their indices
                tmp = STATIC_TILES.copy()
                tmp.update(tilesToRemove)
                tmp.discard(2)                                                                                          # Tile 2 cannot be removed due to codec constraint
                # te2 = time.time()
                # print(tilesToRemove)

                # Removing unwanted tiles
                if len(tmp) == 0:                                                                                       # no tiles removed, send segment as it is
                    self.copyFile(tileVideoName.name, str(data))
                else:
                    f = self.removeTiles(tileVideoName.name, str(data), list(tmp))
                    # print(data,">>>>>>>>>>>>>>>>>", f)
                te3 = time.time()
                if save_runtime:
                    runtimeArr.append(te3-ts1)
                # print(f"Bitrate Extraction time = {te1-ts1}s ; Decision Making time = {te2-te1}s;")
                # print(f'[+] Total time spent on a seg : {te3-ts1}s')

        if save_runtime:
            Path("Runtime").mkdir(parents=True, exist_ok=True)
            jb.dump(np.array(runtimeArr), "Runtime/" + tileVideoName.name + "_runtime.pkl")


if __name__ == "__main__":
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiled-video-dir", type=str, help="Tiled video file name to get tiles dimensions")
    parser.add_argument("--percentile-array-filename", type=str, help="Output array from calibration to use for best percentile calculation")
    parser.add_argument("--cluster-indices-file", type=str, help="Pickle file with index of both clusters")
    parser.add_argument("--cluster-size", default=10, type=int, help="Number of elements in the clusters")
    parser.add_argument("--tiles", default=16, type=int, help="Number of tiles (e.g. 16)")
    parser.add_argument("--gamma", default=1.75, type=float, help="Value of gamma to use for outlier detection")
    parser.add_argument("--save-runtime", action='store_true', help="Whether to store runtime during the run")
    parser.add_argument("--object-ratio-limit", default=0.1, type=float, help="Lower limit of the percentage of objects seen during calibration to enable precision fallback")
    opt = parser.parse_args() 
    print(opt) 

    STATIC_TILES = {2}                                                                                      # Previously decided (by YOLO or other DNN). Tile 2 cannot be removed
    NUMBER_OF_SEGMENTS_DURING_CALIBRATION = 60
    TOTAL_TILES = opt.tiles
    gamma = opt.gamma
    clusterIndicesFile = opt.cluster_indices_file
    clusterSize = opt.cluster_size
    tiledSegmentsDirectory = opt.tiled_video_dir
    percentileArrayFileName = opt.percentile_array_filename                              
    objectRatioLimit = opt.object_ratio_limit
    saveRuntime = opt.save_runtime

    tileClipper = TileClipper(static_tiles = STATIC_TILES, 
                              total_tiles = TOTAL_TILES, 
                              number_of_calibration_segments = NUMBER_OF_SEGMENTS_DURING_CALIBRATION, 
                              object_ratio_limit = objectRatioLimit,
                              cluster_size = clusterSize, 
                              gamma = gamma)
    
    tileClipper.run(tiled_video_segment_folder = tiledSegmentsDirectory,
                    percentile_array_file_from_server = percentileArrayFileName,
                    cluster_indices_file_from_server = clusterIndicesFile,
                    save_runtime = saveRuntime)