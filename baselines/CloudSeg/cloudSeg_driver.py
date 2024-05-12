##################################################################
# Script for CloudSeg
#
# Generates both LR and HR frames in same directory 
# as required by carn. Finaly generates SR frames from 
# LR images and deletes the generated LR and HR frames. Generating
# videos of SR and LR frames in sr_videos_x{scale} folder.
##################################################################

from pathlib import Path
import subprocess as sp
import shutil
# import joblib

FFMPEG = "ffmpeg"

class cloudSeg:
    def __init__(self, scale=2): # scale = 2 or 4; it specifies how much the resolution should be reduced
        self.scale = scale
 
    # Generates low res frames form segmented raw video files
    def generateLRFramesFromVideo(self, inputLRVideo, lrFramesSavePath, frameRate=30):
        videoPath = Path(inputLRVideo)
        # print(str(videoPath))
        Path(lrFramesSavePath+f"/x{self.scale}").mkdir(parents=True, exist_ok=True)
        sp.run([FFMPEG, "-hide_banner", "-loglevel", "quiet", "-i", str(videoPath), "-vf", f"fps={frameRate}", "-vf", f"scale=iw*{str(1/self.scale)}:ih*{str(1/self.scale)}", f"{lrFramesSavePath}/x{self.scale}/frame_%05d_{self.scale}_LR.png"])    # generates .png images in lrFramesSavePath/x{scale}
    
    # Generates high res frames form segmented raw video files
    def generateHRFramesFromVideo(self, inputHRVideo, hrFramesSavePath, frameRate=30):
        videoPath = Path(inputHRVideo)
        # print(str(videoPath))
        Path(hrFramesSavePath+f"/x{self.scale}").mkdir(parents=True, exist_ok=True)
        sp.run([FFMPEG, "-hide_banner", "-loglevel", "quiet", "-i", str(videoPath), "-vf", f"fps={frameRate}", f"{hrFramesSavePath}/x{self.scale}/frame_%05d_{self.scale}_HR.png"])    # generates .png images in hrFramesSavePath/x{scale}

    # generate video from frames (images.png)
    def generateVideoFromFrames(self, framesPath, frameNameFormat, videoSaveDirPath, videoName="out.mp4"): # frameNameFormat = frame_lr_%05d_{self.scale}_HR.png or frame_lr_%05d_{self.scale}_LR.png
        # path = Path(framesPath)
        Path(videoSaveDirPath).mkdir(parents=True, exist_ok=True)        
        # sp.run([FFMPEG, "-hide_banner", "-loglevel", "quiet", "-i", str(framesPath+"/"+frameNameFormat), "-vf", "fps=30", "-c:v", "libx264", "-pix_fmt", "yuv420p", videoSaveDirPath+f"/{videoName}"])
        sp.run([FFMPEG, "-hide_banner", "-loglevel", "quiet", "-framerate", "30" , "-i", str(framesPath+"/"+frameNameFormat), "-c:v", "libx265", "-pix_fmt", "yuv420p", videoSaveDirPath+f"/{videoName}"])
  
    # runs CARN super resolution model on the LR frames
    def runCARN(self, lrFramesPath, cuda=False):
        if cuda == False:
            sp.run(["python3", "CARN/carn/sample.py", "--model", "carn", "--test_data_dir", f"CARN/dataset/{lrFramesPath}", "--scale", f"{self.scale}", "--ckpt_path", "CARN/checkpoint/carn.pth", "--sample_dir", "CARN/sample"])
        else:
            sp.run(["python3", "CARN/carn/sample.py", "--model", "carn", "--test_data_dir", f"CARN/dataset/{lrFramesPath}", "--scale", f"{self.scale}", "--ckpt_path", "CARN/checkpoint/carn.pth", "--sample_dir", "CARN/sample", "--cuda"])
        # return f"CARN/sample/carn/{lrFramesPath.name}/x{self.scale}/SR" # saved super resolution frames path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hr_videodataset_path", type=str, help="High resolution videos.")
    parser.add_argument("--scale", default=2, type=int, help="Scale to use for low resolution videos.")
    parser.add_argument("--cuda", action='store_true', help="Whether to use CUDA support")
    out = parser.parse_args()
    

    cs = cloudSeg(out.scale)
    # file structure => DatasetName/Untiled_mp4_30qp/VideoName/video.mp4
    
    # generating low res & high res frames in CARN/dataset/videoName/segName/x{scale} folder from raw (0.5 sec) video segments
    # Path(f"lr_videos_{out.scale}").mkdir(parents=True, exist_ok=True) 
    for video in sorted(list(Path(out.input_hr_videodataset_path+"/Untiled_mp4_30qp").iterdir())):
        for seg in sorted(list(Path(video).iterdir())):
            # generating frames
            cs.generateLRFramesFromVideo(str(seg), f"CARN/dataset/{video.stem}/{seg.stem}") # saving in dataset folder of CARN
            cs.generateHRFramesFromVideo(str(seg), f"CARN/dataset/{video.stem}/{seg.stem}") # saving in dataset folder of CARN
            if out.cuda:
                cs.runCARN(f"{video.stem}/{seg.stem}", cuda=True)
            else:
                cs.runCARN(f"{video.stem}/{seg.stem}", cuda=False)
            sr_img_path = f"CARN/sample/carn/{seg.stem}/x{cs.scale}/SR"
            lr_img_path = f"CARN/dataset/{video.stem}/{seg.stem}/x{cs.scale}"
            cs.generateVideoFromFrames(framesPath=sr_img_path, frameNameFormat=f"frame_%05d_{cs.scale}_SR.png", videoSaveDirPath=f"sr_videos_x{cs.scale}/{video.stem}", videoName=f"{seg.stem}_SR.mp4") # for SR frames
            cs.generateVideoFromFrames(framesPath=lr_img_path, frameNameFormat=f"frame_%05d_{cs.scale}_LR.png", videoSaveDirPath=f"lr_videos_x{cs.scale}/{video.stem}", videoName=f"{seg.stem}_LR.mp4") # for LR frames
            # shutil.rmtree(f"CARN/dataset/{video.stem}/{seg.stem}/x{cs.scale}")
            # shutil.rmtree(f"CARN/sample/carn/{seg.stem}/x{cs.scale}")
        # shutil.rmtree(f"CARN/dataset/{video.stem}")
        # shutil.rmtree(f"CARN/sample/carn/{seg.stem}")
