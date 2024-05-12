#########################################################################
# This scipt is used for calibration. Requires GPAC.
# It uses the ground-truth file generated using StrongSORT based Yolov5
# to estimate the best percentile for the clusters of tiles by picking 
# the one that gives the best F2-score (emphasizing recall).
# Usage:
#  python calibrate.py --tiled-video-dir dataset/tiled_4x4_mp4/videoName/
#                      --assets-folder ../assets
#########################################################################

from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
import numpy as np
import subprocess as sp
import argparse
import joblib as jb
import warnings
warnings.filterwarnings('error')

class EvictingQueue():
    '''
    Class to implement single ended queue/evicting
    queue. It only provides method to add with no
    dequeue method. The element from the other
    end gets removed automatically on adding
    an element if the queue is full.
    '''
    def __init__(self, elements: list|tuple, size: int = 10) -> None:
        self.size = size
        if len(elements) == size:
            self.__list = [i for i in elements]
        elif len(elements) > size:
            self.__list = [elements[(-size + i)] for i in range(size)]
        else:
            self.__list = [i for i in elements]


    def append(self, element: float|int) -> None:
        """
        Adds elements to end of the queue and removes one 
        element from front if queue size > specified one.
        """
        if type(element) == float or type(element) == int:
            self.__list.append(element)
        else:
            raise TypeError("Elements should of type float or int")
        if len(self.__list) > self.size:
            self.__list.pop(0)


    def __repr__(self) -> str:
        return f"EvictingQueue([{', '.join([str(i) for i in self.__list])}])"


    def __len__(self) -> int:
        return len(self.__list)


    def __iter__(self):
        return iter(self.__list)
    

class Calibrate():
    def __init__(self, tiled_video_directory: str, start_percentile: int = 10, end_percentile: int = 81, number_of_calibration_segments: int = 60, total_tiles: int = 16, cluster_size: int = 10):
        self.tiledVideoDirectory  = tiled_video_directory
        self.startPercentile = start_percentile
        self.endPercentile = end_percentile
        self.totalTiles = total_tiles
        self.clusterSize = cluster_size
        self.numberOfCalibrationSegments = number_of_calibration_segments


    def readGroundTruthFile(self, groundtruth_file_from_yolov5: str) -> np.ndarray:
        """
        Reads the ground-truth text file from Yolov5
        """
        with open(groundtruth_file_from_yolov5, 'r') as file:
            content = file.readlines()
            gtValues = []
            for line in content:
                gtValues.append([int(i) for i in line.split(' ')[0:-1]])

        groundTruth = np.zeros((len(gtValues), self.totalTiles), dtype=bool)
        for i in range(len(gtValues)):
            for j in gtValues[i]:
                groundTruth[i][j-2] = 1

        return groundTruth


    def calculateMetrics(self, selected: np.ndarray, ground_truth_during_calibration: np.ndarray) -> float:
        """
        Calculates F2-score using TP, TN, FP, and FN
        """
        tp = np.sum(np.logical_and(selected, ground_truth_during_calibration))
        fp = np.sum(np.logical_and(selected, np.logical_not(ground_truth_during_calibration)))
        tn = np.sum(np.logical_and(np.logical_not(selected), np.logical_not(ground_truth_during_calibration)))
        fn = np.sum(np.logical_and(np.logical_not(selected), ground_truth_during_calibration))
        try:
            recall = tp / (tp + fn)
        except:
            recall = 1
        try:
            precision = tp / (tp + fp)
        except:
            precision = 1
        try:
            f2 = round((5 * precision * recall)/((4 * precision) + recall), 4)  # finds f2-score to give more importance to recall
        except:
            f2 = 0
        return f2


    def calculateThresholds(self, true_cluster: list, false_cluster: list, bitrates: np.ndarray, lower_percentile: int, upper_percentile: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates threshold and returns the selected video segments
        alogwith the current threshold
        """
        selected = np.zeros(len(bitrates), dtype=bool)
        thresholds = np.zeros(len(bitrates))
        for i in range(len(bitrates)):
            currentThreshold = (np.percentile(list(true_cluster), lower_percentile) + np.percentile(list(false_cluster), upper_percentile)) / 2
            if bitrates[i] > currentThreshold:
                selected[i] = True
                true_cluster.append(float(bitrates[i]))
            else:
                false_cluster.append(float(bitrates[i]))
            thresholds[i] = currentThreshold
        return selected, thresholds


    def getMetrics(self, first_n_true_cluster_indxs: np.ndarray, first_n_false_cluster_indxs: np.ndarray, bitrates_during_calibration: np.ndarray, ground_truth_from_yolov5: np.ndarray, lower_percentile: int, upper_percentile: int) -> float:
        """
        Performs thresholding and returns the perform based on the 
        current threshold
        """
        if len(first_n_true_cluster_indxs) != 0:
            _trueCluster = EvictingQueue(list(bitrates_during_calibration[first_n_true_cluster_indxs]), size = self.clusterSize)
        else:
            _trueCluster = EvictingQueue([max(bitrates_during_calibration)], size = self.clusterSize)

        if len(first_n_false_cluster_indxs) != 0:
            _falseCluster = EvictingQueue(list(bitrates_during_calibration[first_n_false_cluster_indxs]), size = self.clusterSize)
        else:
            _falseCluster = EvictingQueue([min(bitrates_during_calibration)], size = self.clusterSize)
        
        selected, thresholds = self.calculateThresholds(_trueCluster, _falseCluster, bitrates_during_calibration, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
        f2Score = self.calculateMetrics(selected, ground_truth_from_yolov5)
    
        return f2Score


    def getClusterIndices(self, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns indices of true and false clusters
        """
        first_n_true_segments_indx = np.where(ground_truth == True)[0]
        first_n_false_segments_indx = np.where(ground_truth == False)[0]
        return first_n_true_segments_indx, first_n_false_segments_indx


    def main(self, bitrates: np.ndarray, ground_truth_from_yolov5: str, folder_name_to_save_in: str):
        """
        Main driver code to perform grid search over all possible percentiles
        """
        meanFScore, clusterIndices = [], []
        ground_truth = self.readGroundTruthFile(ground_truth_from_yolov5)
        print("Grid searching....")
        for i in tqdm(range(self.startPercentile, self.endPercentile, 10)): # grid searching for right percentile
            for k in range(self.startPercentile, self.endPercentile, 10):   # grid searching for right percentile
                for j in range(self.totalTiles):
                    first_n_true_segments_indx, first_n_false_segments_indx = self.getClusterIndices(ground_truth[:self.numberOfCalibrationSegments, j])
                    if i == self.startPercentile and k == self.startPercentile:
                        clusterIndices.append([first_n_true_segments_indx, first_n_false_segments_indx])
                    f2 = self.getMetrics(first_n_true_segments_indx, first_n_false_segments_indx, bitrates[:self.numberOfCalibrationSegments, j], ground_truth[:self.numberOfCalibrationSegments, j], lower_percentile=i, upper_percentile=k)        
                    meanFScore.append([i, k, j, f2])                        # lower_percentile, upper_percentile, tile_number, f_score

        jb.dump(meanFScore, f"{str(Path(folder_name_to_save_in))}/f2s_{str(Path(self.tiledVideoDirectory).stem)}_cluster{self.clusterSize}.pkl")
        jb.dump(clusterIndices, f"{str(Path(folder_name_to_save_in))}/{str(Path(self.tiledVideoDirectory).stem)}_cluster_indices.pkl")


if __name__ == "__main__":
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiled-video-dir", type=str, help="Tiled video file name to get tiles dimensions")
    parser.add_argument("--start-percentile", default=10, type=float, help="Start percentile for grid searching")
    parser.add_argument("--end-percentile", default=81, type=float, help="End percentile for grid searching")
    parser.add_argument("--cluster-size", default=10, type=int, help="Number of elements in the clusters")
    parser.add_argument("--num-cal-seg", default=60, type=int, help="Number of video segments to use for calibration")
    parser.add_argument("--tiles", default=16, type=int, help="Number of tiles (e.g. 16)")
    parser.add_argument("--assets-folder", default="../", type=str, help="Assets folder where groundtruth is stored.")
    opt = parser.parse_args() 

    cal = Calibrate(tiled_video_directory = opt.tiled_video_dir,
                    start_percentile = opt.start_percentile,
                    end_percentile = opt.end_percentile,
                    number_of_calibration_segments = opt.num_cal_seg,
                    total_tiles = opt.tiles,
                    cluster_size = opt.cluster_size)

    tileVideoName = Path(cal.tiledVideoDirectory)
    tiledSegments = sorted(list(tileVideoName.iterdir()))[:cal.numberOfCalibrationSegments]
    bitrates = np.zeros((cal.numberOfCalibrationSegments, cal.totalTiles))

    # Creating the bitrate array
    for i, segment in enumerate(tiledSegments):
        bitrate = sp.run(["ffprobe", "-v", "error",
                    "-show_entries", "stream=bit_rate",
                    "-of", "default=noprint_wrappers=1", 
                    segment], stdout=sp.PIPE)

        arr = np.fromiter(map(lambda x: int(x[9:]), bitrate.stdout.decode().split('\n')[1:-1]), dtype=int)  # 9 => 'bit_rate='; [1:-1] => 1 because tile 1 has metadata only

        bitrates[i] = arr
    print("Created bitrate array")

    groundTruthFile = f"{opt.assets_folder}/GroundTruths_TileLevel/{tileVideoName.stem}.txt"
    outFolder = Path(f"{opt.assets_folder}/F2s/").mkdir(parents=True, exist_ok=True)
    
    # Running grid search
    cal.main(bitrates, groundTruthFile, f"{opt.assets_folder}/F2s/")
