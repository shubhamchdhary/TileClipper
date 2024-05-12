# Script to get Accuracy, precision, recall, FP, etc of bitrate based detection using 

from collections import Counter
from pathlib import Path
import numpy as np

class CaculateMetrics():
    def __init__(self, datasetName, filename, method="TileClipper", weather=None, numCalibrationSegs=60, numFamesInASeg=15, clusterSize=10, tiles="4x4", percentilesForClusters_l=None, percentilesForClusters_h=None): # weather is used in case of recalibration, weather = "Dawn" when recalibrated at dawn and "Noon" when recalibrated at noon
        self.datasetName = datasetName
        self.filename = filename
        self.method = method
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0 # True Positives, True Negatives, False Positives, False Negatives
        self.QP_Savings = 0
        self.actual_saving = 0 
        self.total_saving = 0
        self.g_data, self.t_data = 0, 0
        self.f1 = 0
        self.precision = 0
        self.recall = 0
        self.weather = weather
        self.BUFFERED_SEGS = 0
        self.CALIBRATION_BUFFER = numCalibrationSegs
        self.NUMBER_OF_FRAMES_IN_A_SEG = numFamesInASeg
        self.clusterSize = clusterSize
        self.tiles = tiles
        self.percentilesForClusters_l = percentilesForClusters_l
        self.percentilesForClusters_h = percentilesForClusters_h
        # print(self.method, "Constructor")

    def reset(self):
        self.TP, self.TN, self.FP, self.FN  = 0, 0, 0, 0

    def calc_metrics_individual(self, groundtruth_file, test_file, frames_in_a_seg, total_buffered_segs):
        # Reading ground truth file
        with open(groundtruth_file, 'r') as fg:
            g_d = fg.read()
        self.g_data = [i.split(" ")[:-1] for i in g_d.split('\n')[0:-1]]

        # Reading test file
        with open(test_file, 'r')as ft:
            t_d = ft.read()
        self.t_data = [i.split(" ")[:-1] for i in t_d.split('\n')[0:-1]]

        segs = 0
        accu, f1, prec, recal = [], [], [], []
        for frame_indx in range(len(self.t_data)):
            g_frame_objs = Counter(self.g_data[frame_indx + (total_buffered_segs*frames_in_a_seg)])  # number of occurences of each object class
            t_frame_objs = Counter(self.t_data[frame_indx])
            # print(g_frame_objs, t_frame_objs)
            for cls in range(8):   # loop for 0 to 7 unique classes
                if str(cls) in g_frame_objs and str(cls) in t_frame_objs:
                    if g_frame_objs[cls] > t_frame_objs[str(cls)]:
                        self.TP += t_frame_objs[str(cls)]
                        self.FN += g_frame_objs[str(cls)] - t_frame_objs[str(cls)]
                    elif g_frame_objs[str(cls)] < t_frame_objs[str(cls)]:
                        self.TP += g_frame_objs[str(cls)]
                        self.FP += t_frame_objs[str(cls)] - g_frame_objs[str(cls)]    
                    else:
                        self.TP += g_frame_objs[str(cls)]

                elif str(cls) not in g_frame_objs and str(cls) not in t_frame_objs:
                    self.TN += 1 

                elif str(cls) not in g_frame_objs and str(cls) in t_frame_objs:
                    self.FP += t_frame_objs[str(cls)]

                elif str(cls) in g_frame_objs and str(cls) not in t_frame_objs:
                    self.FN += g_frame_objs[str(cls)]
            if (frame_indx+1)%(frames_in_a_seg) == 0: # number of frames in a 0.5sec segment
                # print(self.TP, self.TN, self.FP, self.FN)
                a, f, p, r = self.get_metric(self.TP, self.TN, self.FP, self.FN)
                accu.append(a); f1.append(f); prec.append(p); recal.append(r)
                self.reset()
                segs += 1
        accu, f1, prec, recal = np.array(accu), np.array(f1), np.array(prec), np.array(recal)
        self.reset()
        return np.mean(accu), np.mean(f1), np.mean(prec), np.mean(recal), np.std(accu), np.std(f1), np.std(prec), np.std(recal)


    def calc_metrics(self): # name of mp4 file without extension, type = "TileClipper" or "Static" or "Cloudseg" or "DDS" or "Reducto"
        # groundTruthFile_labels = f"../../NSDI/Ground_Truths_yolov5s/{self.datasetName}/{self.filename}.txt"
        groundTruthFile_labels = f"../assets/GroundTruths/{self.filename}.txt"
        if self.method == "TileClipper":
            if self.weather == None:
                testFile_labels = f"../assets/labels/TileClipper/{self.filename}.txt"
            # elif self.weather == "Dawn":
                # testFile_labels = f"../assets/labels/TileClipper/TileClipper_Without_Recalibration/When_calibrated_at_dawn/{self.filename}.txt"
            elif self.weather == "Noon":
                testFile_labels = f"../assets/labels/TileClipper/TileClipper_Without_Recalibration/When_calibrated_at_noon/{self.filename}_recalib.txt"
        elif self.method == "Static":
            testFile_labels = f"../assets/labels/StaticallyRemovedLabels/{self.filename}.txt"
        elif self.method == "CloudSeg":
            testFile_labels = f"../assets/labels/CloudSegLabels/{self.filename}.txt"
        elif self.method == "DDS":
            # self.filename = str(self.filename)[:-4]+"30qp"
            testFile_labels = f"../assets/labels/DDSLabels/{self.filename}.txt"

        # Without Calibration       
        elif self.method == "Without_Calibration":
            testFile_labels = f"../assets/labels/Ablation_Study/Without_Calibration/{self.percentilesForClusters_l}per/{self.filename}.txt"

        # Without fallback
        elif self.method == "without_fallback":
            testFile_labels = f"../assets/labels/Ablation_Study/without_fallback/{self.filename}.txt"

        # Sensitivity tests
        elif self.method == "differentBufferSizes":
            testFile_labels = f"../assets/labels/SensitivityTests/differentBufferSizes/{self.clusterSize}_buff/{self.filename}.txt"
        elif self.method == "differentCalibrationSegments":
            testFile_labels = f"../assets/labels/SensitivityTests/differentCalibrationSegments/{self.CALIBRATION_BUFFER}_calSegs/{self.filename}.txt"
        elif self.method == "differentTileConf":
            testFile_labels = f"../assets/labels/SensitivityTests/differentTileConf/{self.tiles}/{self.filename}.txt"

        # print(testFile_labels)
        BUFFERED_SEGS = self.BUFFERED_SEGS
        CALIBRATION_BUFFER = self.CALIBRATION_BUFFER
        NUMBER_OF_FRAMES_IN_A_SEG = self.NUMBER_OF_FRAMES_IN_A_SEG

        # Reading ground truth file
        with open(groundTruthFile_labels, 'r') as fg:
            g_d = fg.read()
        self.g_data = [i.split(" ")[:-1] for i in g_d.split('\n')[0:-1]]
        # print(self.g_data)

        # Reading test file
        with open(testFile_labels, 'r')as ft:
            t_d = ft.read()
        self.t_data = [i.split(" ")[:-1] for i in t_d.split('\n')[0:-1]]

        # print(len(self.g_data), len(self.t_data), CALIBRATION_BUFFER, self.filename, "><><><><><><><")

        if self.method != "Static":
            # print(">>>>>>>>>>>", self.method, "Metric")
            # Performance metric calculation ################################
            ours = ["TileClipper", "Without_Calibration", "without_fallback", "differentBufferSizes", "differentCalibrationSegments", "differentTileConf"]
            # if self.method == "TileClipper" or self.method == "Without_Calibration" or self.method == "without_fallback":
            if self.method in ours:
                # print(f"[d] g_data {len(self.g_data)-(BUFFERED_SEGS+CALIBRATION_BUFFER)*NUMBER_OF_FRAMES_IN_A_SEG}; t_data {len(self.t_data)}")
                segs = 0
                accu, f1, prec, recal = 0, 0, 0, 0
                for frame_indx in range(len(self.t_data)):
                    g_frame_objs = Counter(self.g_data[frame_indx + ((BUFFERED_SEGS+CALIBRATION_BUFFER)*NUMBER_OF_FRAMES_IN_A_SEG)])  # number of occurences of each object class
                    t_frame_objs = Counter(self.t_data[frame_indx])
                    for cls in range(8):   # loop for 0 to 7 unique classes
                        if str(cls) in g_frame_objs and str(cls) in t_frame_objs:
                            if g_frame_objs[cls] > t_frame_objs[str(cls)]:
                                self.TP += t_frame_objs[str(cls)]
                                self.FN += g_frame_objs[str(cls)] - t_frame_objs[str(cls)]
                            elif g_frame_objs[str(cls)] < t_frame_objs[str(cls)]:
                                self.TP += g_frame_objs[str(cls)]
                                self.FP += t_frame_objs[str(cls)] - g_frame_objs[str(cls)]    
                            else:
                                self.TP += g_frame_objs[str(cls)]

                        elif str(cls) not in g_frame_objs and str(cls) not in t_frame_objs:
                            self.TN += 1 

                        elif str(cls) not in g_frame_objs and str(cls) in t_frame_objs:
                            self.FP += t_frame_objs[str(cls)]

                        elif str(cls) in g_frame_objs and str(cls) not in t_frame_objs:
                            self.FN += g_frame_objs[str(cls)]
                    if (frame_indx+1)%15 == 0: # number of frames in a 0.5sec segment
                        # print(self.TP, self.TN, self.FP, self.FN)
                        a, f, p, r = self.get_metric(self.TP, self.TN, self.FP, self.FN)
                        accu+=a; f1+=f; prec+=p; recal+=r
                        self.reset()
                        segs += 1
                # print(self.filename, accu, f1, prec, recal, segs, ">>>>>>")
                accu, f1, prec, recal = accu/segs, f1/segs, prec/segs, recal/segs
                self.reset()

            else:
                # print(f"[d] g_data {len(self.g_data)}; t_data {len(self.t_data)}")
                segs = 0
                accu, f1, prec, recal = 0, 0, 0, 0
                for frame_indx in range(len(self.t_data) - ((BUFFERED_SEGS+CALIBRATION_BUFFER)*NUMBER_OF_FRAMES_IN_A_SEG)):
                    g_frame_objs = Counter(self.g_data[frame_indx + ((BUFFERED_SEGS+CALIBRATION_BUFFER)*NUMBER_OF_FRAMES_IN_A_SEG)])   # number of occurences of each object class
                    t_frame_objs = Counter(self.t_data[frame_indx + ((BUFFERED_SEGS+CALIBRATION_BUFFER)*NUMBER_OF_FRAMES_IN_A_SEG)])
                    for cls in range(8):   # loop for 0 to 7 unique classes
                        if str(cls) in g_frame_objs and str(cls) in t_frame_objs:
                            if g_frame_objs[cls] > t_frame_objs[str(cls)]:
                                self.TP += t_frame_objs[str(cls)]
                                self.FN += g_frame_objs[str(cls)] - t_frame_objs[str(cls)]
                            elif g_frame_objs[str(cls)] < t_frame_objs[str(cls)]:
                                self.TP += g_frame_objs[str(cls)]
                                self.FP += t_frame_objs[str(cls)] - g_frame_objs[str(cls)]    
                            else:
                                self.TP += g_frame_objs[str(cls)]

                        elif str(cls) not in g_frame_objs and str(cls) not in t_frame_objs:
                            self.TN += 1 

                        elif str(cls) not in g_frame_objs and str(cls) in t_frame_objs:
                            self.FP += t_frame_objs[str(cls)]

                        elif str(cls) in g_frame_objs and str(cls) not in t_frame_objs:
                            self.FN += g_frame_objs[str(cls)]
                    if (frame_indx + 1)%15 == 0: # number of frames in a 0.5sec segment
                        a, f, p, r = self.get_metric(self.TP, self.TN, self.FP, self.FN)
                        accu+=a; f1+=f; prec+=p; recal+=r
                        self.reset()
                        segs += 1
                # self.TP, self.TN, self.FP, self.FN = 1, 1, 0, 0
                accu, f1, prec, recal = accu/segs, f1/segs, prec/segs, recal/segs
                self.reset()

        else:
            self.TP, self.TN, self.FP, self.FN = 1, 1, 0, 0
            accu, f1, prec, recal = self.get_metric(self.TP, self.TN, self.FP, self.FN)
            self.reset()

        return accu, f1, prec, recal

    def print_difference(self):
        print(f"[d] g_data {len(self.g_data)}; t_data {len(self.t_data)}")

    def calc_saving_individual(self, tiled_video_folder_path, removed_tile_folder_path, buffered_segs):
        tiled_mp4_size = []
        pp = sorted(list(Path(tiled_video_folder_path).iterdir()))
        for ii in pp[buffered_segs:]: # first 20 segs buffered 
            tiled_mp4_size.append(Path(ii).stat().st_size/1024)

        removed_tile_mp4_size = []
        pp = Path(removed_tile_folder_path).iterdir()
        for ii in pp:
            removed_tile_mp4_size.append(Path(ii).stat().st_size/1024)

        tiled_mp4_size = np.array(tiled_mp4_size)
        removed_tile_mp4_size = np.array(removed_tile_mp4_size)
        indiviSavings = ((tiled_mp4_size - removed_tile_mp4_size)/tiled_mp4_size)*100
        self.QP_Savings = 0
        self.actual_saving = np.mean(indiviSavings)
        self.total_saving = np.mean(indiviSavings + self.QP_Savings)
        return self.QP_Savings, self.actual_saving,  np.std(indiviSavings), self.total_saving

    def calc_calibration_overhead(self):
        # Retuns overhead on KB
        # print(">>>>>>>>>>>", self.method)
        _path = "../../../"
        overhead = 0
        # baselinePath = "../../Videos/"

        # Saving Calculation ############################################
        # print(">>>>>>>>>>>", self.method, "<<<<<<<<<<<<<"
        if self.method == "Without_Calibration":
            # print("Saving", self.method)
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"Videos/Without_Calibration/tiled_4x4_mp4/{self.filename}").iterdir()))[:(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER)]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            # removed_tile_mp4_size = 0
            # pp = Path(_path+f"Videos/Without_Calibration/removedTileMp4/{self.filename}").iterdir()
            # for ii in pp:
            #     removed_tile_mp4_size += Path(ii).stat().st_size/1024
            # TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            overhead = reduced_tiled_mp4_size

        elif self.method == "without_fallback":
            # print("Saving", self.method)
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"Videos/without_fallback/tiled_4x4_mp4/{self.filename}").iterdir()))[:(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER)]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            # removed_tile_mp4_size = 0
            # pp = Path(_path+f"Videos/without_fallback/removedTileMp4/{self.filename}").iterdir()
            # for ii in pp:
            #     removed_tile_mp4_size += Path(ii).stat().st_size/1024                    
            # TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100

            overhead = reduced_tiled_mp4_size

        # Sensitivity tests
        elif self.method == "differentBufferSizes":
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"Videos/SensitivityTests/differentBufferSizes/tiled_4x4_mp4/{self.filename}").iterdir()))[:(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER)]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            # removed_tile_mp4_size = 0
            # pp = Path(_path+f"Videos/SensitivityTests/differentBufferSizes/removedTileMp4/{self.clusterSize}_buff/{self.filename}").iterdir()
            # for ii in pp:
            #     removed_tile_mp4_size += Path(ii).stat().st_size/1024                    
            # TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            overhead = reduced_tiled_mp4_size
        elif self.method == "differentCalibrationSegments":
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"Videos/SensitivityTests/differentCalibrationSegments/tiled_4x4_mp4/{self.filename}").iterdir()))[:(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER)]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            # removed_tile_mp4_size = 0
            # pp = Path(_path+f"Videos/SensitivityTests/differentCalibrationSegments/removedTileMp4/{self.CALIBRATION_BUFFER}_calSegs/{self.filename}").iterdir()
            # for ii in pp:
            #     removed_tile_mp4_size += Path(ii).stat().st_size/1024                    
            # TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            overhead = reduced_tiled_mp4_size

        return 0, overhead, 0 # 0 to make consistent with calc_saving() function


    def calc_savings(self):
        # print(">>>>>>>>>>>", self.method)
        _path = "../"
        baselinePath = "../"
        baselines = ["Static", "CloudSeg", "DDS"]

        if self.method in baselines:
            # Saving got by 30qp encoding
            # untiled_22qpmp4_size = 0 # 22qp
            # pp = Path(_path+f"videos/{self.datasetName}/Untiled_mp4_22qp/{self.filename}").iterdir()
            # for ii in pp:
            #     untiled_22qpmp4_size += Path(ii).stat().st_size/1024

            tiled_mp4_size = 0 # 30qp
            pp = sorted(list(Path(_path+f"videos/{self.datasetName}/tiled_4x4_mp4/{self.filename}").iterdir()))
            for ii in pp:
                tiled_mp4_size += Path(ii).stat().st_size/1024

            # tiled_mp4_calibration_size = 0 # 30qp
            # for ii in pp[20:81]: # 20 segs are just buffered and rest 20 to 80 are sent for calibration
            #     tiled_mp4_calibration_size += Path(ii).stat().st_size/1024

            untiled_30qpmp4_size = 0 # 30qp
            pp = Path(_path+f"videos/{self.datasetName}/Untiled_mp4_30qp/{self.filename}").iterdir()
            for ii in pp:
                untiled_30qpmp4_size += Path(ii).stat().st_size/1024
            
            untiled_30qpmp4_reduced_size = 0 # 30qp
            pp = sorted(list(Path(_path+f"videos/{self.datasetName}/Untiled_mp4_30qp/{self.filename}").iterdir()))
            for ii in pp:
                untiled_30qpmp4_reduced_size += Path(ii).stat().st_size/1024

            # self.QP_Savings = ((untiled_22qpmp4_size - untiled_30qpmp4_size)/untiled_22qpmp4_size)*100
            self.QP_Savings = 0

        # Saving Calculation ############################################
        # print(">>>>>>>>>>>", self.method, "<<<<<<<<<<<<<")
        if self.method == "TileClipper":
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"videos/{self.datasetName}/tiled_4x4_mp4/{self.filename}").iterdir()))[(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER):]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            if self.weather == None:
                removed_tile_mp4_size = 0
                pp = Path(_path+f"videos/{self.datasetName}/removedTileMp4/{self.filename}").iterdir()
                for ii in pp:
                    removed_tile_mp4_size += Path(ii).stat().st_size/1024
            # elif self.weather == "Dawn":
            #     removed_tile_mp4_size = 0
            #     pp = Path(_path+f"videos/TileClipper_Without_Recalibration/When_calibrated_at_dawn/{self.datasetName}/removedTileMp4/{self.filename}").iterdir()
            #     for ii in pp:
            #         removed_tile_mp4_size += Path(ii).stat().st_size/1024
            elif self.weather == "Noon":
                removed_tile_mp4_size = 0
                pp = Path(_path+f"videos/TileClipper_Without_Recalibration/When_calibrated_at_noon/{self.datasetName}/removedTileMp4/{self.filename}").iterdir()
                for ii in pp:
                    removed_tile_mp4_size += Path(ii).stat().st_size/1024

            TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            # self.QP_Savings = 0
            self.actual_saving = TileClipper_Savings


        elif self.method == "Without_Calibration":
            # print("Saving", self.method)
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"videos/Without_Calibration/tiled_4x4_mp4/{self.filename}").iterdir()))[(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER):]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            removed_tile_mp4_size = 0
            pp = Path(_path+f"videos/Without_Calibration/removedTileMp4/{self.percentilesForClusters_l}per/{self.filename}").iterdir()
            for ii in pp:
                removed_tile_mp4_size += Path(ii).stat().st_size/1024
            TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            self.actual_saving = TileClipper_Savings
            self.QP_Savings, self.total_saving = 0, 0

        elif self.method == "without_fallback":
            # print("Saving", self.method)
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"videos/without_fallback/tiled_4x4_mp4/{self.filename}").iterdir()))[(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER):]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            removed_tile_mp4_size = 0
            pp = Path(_path+f"videos/without_fallback/removedTileMp4/{self.filename}").iterdir()
            for ii in pp:
                removed_tile_mp4_size += Path(ii).stat().st_size/1024                    
            TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            self.actual_saving = TileClipper_Savings
            self.QP_Savings, self.total_saving = 0, 0

        # Sensitivity tests
        elif self.method == "differentBufferSizes":
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"videos/SensitivityTests/differentBufferSizes/tiled_4x4_mp4/{self.filename}").iterdir()))[(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER):]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            removed_tile_mp4_size = 0
            pp = Path(_path+f"videos/SensitivityTests/differentBufferSizes/removedTileMp4/{self.clusterSize}_buff/{self.filename}").iterdir()
            for ii in pp:
                removed_tile_mp4_size += Path(ii).stat().st_size/1024                    
            TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            self.actual_saving = TileClipper_Savings
            self.QP_Savings, self.total_saving = 0, 0
        elif self.method == "differentCalibrationSegments":
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"videos/SensitivityTests/differentCalibrationSegments/tiled_4x4_mp4/{self.filename}").iterdir()))[(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER):]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            removed_tile_mp4_size = 0
            pp = Path(_path+f"videos/SensitivityTests/differentCalibrationSegments/removedTileMp4/{self.CALIBRATION_BUFFER}_calSegs/{self.filename}").iterdir()
            for ii in pp:
                removed_tile_mp4_size += Path(ii).stat().st_size/1024                    
            TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            self.actual_saving = TileClipper_Savings
            self.QP_Savings, self.total_saving = 0, 0
        elif self.method == "differentTileConf":
            reduced_tiled_mp4_size = 0 # from segment num calibration segs onwards from tiled_4x4_mp4
            pp = sorted(list(Path(_path+f"videos/SensitivityTests/differentTileConf/tiled_4x4_mp4/{self.tiles}/{self.filename}").iterdir()))[(self.BUFFERED_SEGS + self.CALIBRATION_BUFFER):]
            for ii in pp:
                reduced_tiled_mp4_size += Path(ii).stat().st_size/1024 # in KB

            removed_tile_mp4_size = 0
            pp = Path(_path+f"videos/SensitivityTests/differentTileConf/removedTileMp4/{self.tiles}/{self.filename}").iterdir()
            for ii in pp:
                removed_tile_mp4_size += Path(ii).stat().st_size/1024                    
            TileClipper_Savings = ((reduced_tiled_mp4_size - removed_tile_mp4_size)/reduced_tiled_mp4_size)*100
            # self.total_saving = TileClipper_Savings + self.QP_Savings
            self.actual_saving = TileClipper_Savings
            self.QP_Savings, self.total_saving = 0, 0    

        elif self.method == "Static":
            if Path(baselinePath + f"baselines/StaticTileRemoval/StaticallyRemovedLabels/{self.filename}.txt").exists():
                removed_tile_mp4_size = 0
                pp = Path(_path + f"videos/Baselines/StaticTileRemoval/StaticallyRemovedTiles/{self.datasetName}/{self.filename}").iterdir()
                for ii in pp:
                    removed_tile_mp4_size += Path(ii).stat().st_size/1024

                static_Savings = ((tiled_mp4_size - removed_tile_mp4_size)/tiled_mp4_size)*100
                self.total_saving = static_Savings + self.QP_Savings
                self.actual_saving = static_Savings
            else:
                self.actual_saving = 0
                self.total_saving = self.actual_saving + self.QP_Savings

        elif self.method == "CloudSeg":
            lr_mp4_seg_size = 0
            pp = Path(baselinePath + f"videos/Baselines/CloudSeg/lr_videos_x2/{self.datasetName}/{self.filename}").iterdir()
            for ii in pp:
                lr_mp4_seg_size += Path(ii).stat().st_size/1024

            lr_cs_Savings = ((untiled_30qpmp4_size - lr_mp4_seg_size)/untiled_30qpmp4_size)*100
            self.total_saving = lr_cs_Savings + self.QP_Savings
            self.actual_saving = lr_cs_Savings

        elif self.method == "DDS":
            dds_mp4_seg_size = 0
            # pp = sorted(list(Path(f"Baselines/DDS/FirstPhaseVideos/{self.datasetName}/{self.filename}").iterdir()))[80:]
            pp = sorted(list(Path(baselinePath + f"videos/Baselines/DDS/FirstPhaseVideos/{self.datasetName}/{self.filename}").iterdir()))
            for ii in pp:
                dds_mp4_seg_size += Path(ii).stat().st_size/1024

            dds_Savings = ((untiled_30qpmp4_size - dds_mp4_seg_size)/untiled_30qpmp4_size)*100
            # dds_Savings = ((untiled_30qpmp4_reduced_size - dds_mp4_seg_size)/untiled_30qpmp4_reduced_size)*100
            self.total_saving = dds_Savings + self.QP_Savings
            self.actual_saving = dds_Savings

        return self.QP_Savings, self.actual_saving, self.total_saving

    def get_metric(self, tp, tn, fp, fn):
        if tp != 0:
            precision = (tp)/(tp+fp)
            recall = (tp)/(tp+fn)
            f1_score = 2*((precision*recall)/(precision+recall))
        else:
            precision, recall = 0, 0
            f1_score = 0
        accu = (tp+tn)/(tp+tn+fp+fn)
        return accu, f1_score, precision, recall

    # def print_metric(self):
    #     print(f"[+] QP Savings = {self.QP_Savings} %")
    #     print(f"[+] {self.method} Savings = {self.actual_saving} %")
    #     print(f"[+] Total Savings = {self.total_saving} %")
    #     print(f"[i] _TP = {self.TP}; _TN = {self.TN}; _FP = {self.FP}; _FN = {self.FN}")
    #     print(f"[+] Accuracy = {(self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)}")
    #     print(f"[+] Selectivity (_TNR) = {(self.TN)/(self.TN+self.FP)}")    # To test how many TN are getting detected
    #     print(f"[+] Relative _FN = {(self.FN)/(self.TP+self.TN+self.FP+self.FN)}")
    #     print(f"[+] Relative _FP = {(self.FP)/(self.TP+self.TN+self.FP+self.FN)}")

# if __name__ == "__main__":
#     # cal_metrics(self.datasetName = "IndianVideos", filenameself. = "Day2_5min_") 
#     cal_metrics(datasetName = "DETRAC", filename = "MVI_39761", self.method="static") 