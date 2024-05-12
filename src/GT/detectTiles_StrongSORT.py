import argparse
import IoUBasedMotionTracker_StrongSORT
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from ffmpeg import _probe
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
try:
    from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
except:
    import sys
    sys.path.append('yolov5/utils')
    from dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        count=False,  # get counts of every obhects
        draw=False,  # draw object trajectory lines
        tiled_video=None,
        tiles=16,
        save_labelfolder_name="myLabel/"
):

    source = str(source)
    source = str(source)
    _p = Path(source)
    if _p.is_dir():
        vd = _p.name
    else:
        raise TypeError(f"{source} must be a directory")    
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # def makeDetect(video, numTiles): # video = tiled video, numTiles = number of tiles
    #     d1, d2 = 6, 4
    #     if numTiles == 64:
    #         d1, d2 = 10, 8
    #     vid = _probe.probe(video)
    #     _detect = {}
    #     f = 1
    #     w, h = 0, vid['streams'][1]['height']
    #     _w, _h = vid['streams'][0]['width'], vid['streams'][0]['height']
    #     for i in range(1, numTiles+1):
    #         f += 1
    #         ww = vid['streams'][i]['width']
    #         hh = vid['streams'][i]['height']
    #         w += ww
    #     #     h += hh
    #         if f%d1 == 0: #f%10==0 for 64 tiles (+2 in number of tiles in a row). 6 for 16 tiles.
    #             h = h+hh
    #             f = 2
    #         _detect.update({i+1:[w, h, ww, hh, False]})
    #         if i%d2 == 0: # i%8==0 for 64 tiles, 4 for 16 tiles.
    #             w = 0 
    #     return _detect, _w, _h

    def makeDetect(video, numTiles): # video = tiled video, numTiles = number of tiles
        # d1, d2 = 6, 4
        # if numTiles == 64:
        #     d1, d2 = 10, 8
        _tilesInOneDimension = np.sqrt(numTiles)
        d1, d2 = (_tilesInOneDimension + 2), _tilesInOneDimension   # Used to check index of tiles in y and x dims
        vid = _probe.probe(video)
        _detect = {}
        f = 1
        w, h = 0, vid['streams'][1]['height']
        _w, _h = vid['streams'][0]['width'], vid['streams'][0]['height']
        for i in range(1, numTiles+1):
            f += 1
            ww = vid['streams'][i]['width']
            hh = vid['streams'][i]['height']
            w += ww
            if f%d1 == 0: # Starting new row so reset f and update h
                h = h+hh
                f = 2     # 2 because indexing in tiled video of GPAC starts with 2
            _detect.update({i+1:[w, h, ww, hh, False]})
            if i%d2 == 0: # Reached last column so reset w
                w = 0 
        return _detect, _w, _h
    
    detect_1, _w, _h = makeDetect(tiled_video, tiles) #, 64)
    segment = 0
    iouTracker = IoUBasedMotionTracker_StrongSORT.IoUTracker(np.sqrt(tiles))

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        totalFrames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(">>>>>>>>>>>>>>>", frame_idx, totalFrames)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # video file
            if source.endswith(VID_FORMATS):
                txt_file_name = p.stem
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                txt_file_name = p.parent.name  # get folder name containing current img
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    frameIndex = (frame_idx % 15) + 1
                    # print(frameIndex, frame_idx, totalFrames, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    if frameIndex > 2 and frameIndex < totalFrames:
                        _detections = []
                        # for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        for (output, conf) in zip(outputs[i], confs):
                            _detections.append([output[:4], output[4]])
                        iouTracker.store(_detections)
                        if frameIndex == totalFrames - 1:
                            lst = iouTracker.getMobileObjects()
                            iouTracker.updateDetect(lst, detect_1)
                    elif frameIndex == totalFrames:
                        # at the last frame, writing the file what tiles to keep corresonponding to that segement
                        segment += 1
                        with open(save_labelfolder_name + vd + ".txt", 'a') as t:
                            for d in detect_1:
                                # detect = True means motion was there so keep that tile
                                if detect_1[d][4]==True:
                                    t.write(str(d) + " ")
                                    detect_1[d][4]=False
                            t.write("\n")
                            #print("Reset")
                            # print("Segment: ", segment, s.split(" ")[1])
                            iouTracker.reset()
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        bbox_left, bbox_top, bbox_right, bbox_bottom = bboxes                                                             

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                else:
                    frameIndex = (frame_idx % 15) + 1
                    # print(f'My No detections len == {len(outputs[i])}, {frameIndex} {frame_idx} {segment} {s.split(" ")[1]} >>>>>>>')
                    
                    if frameIndex == totalFrames - 1:
                        lst = iouTracker.getMobileObjects()
                        iouTracker.updateDetect(lst, detect_1) #, _w, _h)
                        # print(">>>>>>>>>>>>>>> len == 0 detect updated", frameIndex, len(iouTracker.lastSeenObjects), len(iouTracker.trackedObjects))
                    elif frameIndex == totalFrames:
                        # at the last frame, writing the file what tiles to keep corresonponding to that segement
                        with open(save_labelfolder_name + vd + ".txt", 'a') as t:
                            for d in detect_1:
                                # detect = True means motion was there so keep that tile
                                if detect_1[d][4]==True:
                                    t.write(str(d) + " ")
                                    detect_1[d][4]=False
                            t.write("\n")
                            segment += 1
                            # print(len(iouTracker.trackedObjects), len(iouTracker.lastSeenObjects))
                            # print("Reset", " Segment: in len", segment, s.split(" ")[1])
                            iouTracker.reset()                
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                frameIndex = (frame_idx % 15) + 1
                LOGGER.info(f'No detections {frameIndex}, {frame_idx}')
                # print(f'No detections {frameIndex}, {frame_idx}')
                if frameIndex == totalFrames - 1:
                    lst = iouTracker.getMobileObjects()
                    iouTracker.updateDetect(lst, detect_1) #, _w, _h)
                    # print(">>>>>>>>>>>>>>> detect updated", frameIndex, len(iouTracker.lastSeenObjects), len(iouTracker.trackedObjects))
                elif frameIndex == totalFrames:
                    # at the last frame, writing the file what tiles to keep corresonponding to that segement
                    with open(save_labelfolder_name + vd + ".txt", 'a') as t:
                        for d in detect_1:
                            # detect = True means motion was there so keep that tile
                            if detect_1[d][4]==True:
                                t.write(str(d) + " ")
                                detect_1[d][4]=False
                        t.write("\n")
                        segment += 1
                        # print(len(iouTracker.trackedObjects), len(iouTracker.lastSeenObjects))
                        # print("Reset", " Segment: ", segment, s.split(" ")[1])
                        iouTracker.reset()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--count', action='store_true', help='display all MOT counts results on screen')
    parser.add_argument('--draw', action='store_true', help='display object trajectory lines')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument("--tiled-video", type=str, help="Tiled video file name.")
    parser.add_argument("--tiles", default=16, type=int, help="Number of tiles in the tiled video")
    parser.add_argument('--save-labelfolder-name', type=str, default="myLabel/", help='Folder (with "/") where to save label.txt file')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
