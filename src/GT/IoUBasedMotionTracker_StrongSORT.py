"""
This is file to provide support for groud truth generation by
tracking and removing the static objects. It helps detect the
mobile objects. To track the vehicles it uses pre-existing 
StrongSORT + Yolo implementation tracker code from 
https://github.com/bharath5673/StrongSORT-YOLO/tree/main. 
To find static objects, it measures the IoU of its bounding boxes
in the first and last frame. If it is very high (>0.8), then the 
object is static.
"""

import numpy as np

class BBox():
    """Bounding Box"""
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def calcArea(self) -> int:
        """Returns area of the BBox"""
        return self.w * self.h
    

class IoUCalculator():
    """Calculates IoU from BBoxes""" 
    def __init__(self, bbox_1: BBox, bbox_2: BBox):
        self.bbox_1 = bbox_1
        self.bbox_2 = bbox_2
        
    def calcIntersection(self) -> int:
        """Calculates intersection of Bounding Boxes"""
        overlapX = max(0, min((self.bbox_1.x + self.bbox_1.w), (self.bbox_2.x + self.bbox_2.w)) - max((self.bbox_1.x), (self.bbox_2.x)))
        overlapY = max(0, min((self.bbox_1.y + self.bbox_1.h), (self.bbox_2.y + self.bbox_2.h)) - max((self.bbox_1.y), (self.bbox_2.y)))
        return overlapX * overlapY

    def calcUnion(self) -> int:
        """Calculates Union of Bounding Boxes"""
        return ((self.bbox_1.calcArea() + self.bbox_2.calcArea()) - self.calcIntersection())

    def calcIoU(self) -> float:
        """Calculates IoU"""
        return self.calcIntersection()/self.calcUnion()


class IoUTracker():
    """Find static and mobile objects using IoU of BBoxes"""
    def __init__(self, tilesInOneDimension: int):
        self.trackedObjects = {}
        self.lastSeenObjects = {}
        self.tilesInOneDimension = tilesInOneDimension

    def store(self, objects_bbox: list) -> np.array:
        """ Track BBoxes using StrongSORT
            object_bbox (list): [xywh, conf, class, img0]    
        """ 
        for rect in objects_bbox:
            x1, y1, x2, y2 = rect[0]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            id = int(rect[1])
            # print("x, y, w, h, id >>>>>>>>>>", x, y, w, h, id, "x1, y1, x2, y2 >>>>>>>>>>", x1, y1, x2, y2)
            if len(self.trackedObjects) == 0:
                # Storing objects of the first frame
                self.trackedObjects.update({id : [x, y, w, h]})
                self.lastSeenObjects.update({id : [x, y, w, h]})
                # print(self.trackedObjects)
            else:
                if id not in self.trackedObjects:
                    self.trackedObjects.update({id : [x, y, w, h]})
                if id not in self.lastSeenObjects:
                    self.lastSeenObjects.update({id : [x, y, w, h]})
                else:
                    self.lastSeenObjects[id] = [x, y, w, h]

    def getMobileObjects(self, static_object_iou_threshold: float=0.8) -> list:
        """
        Find and remove static objects
        """
        lst = []
        for _id in self.lastSeenObjects: 
            rect = self.lastSeenObjects[_id]
            x, y, w, h = rect[0], rect[1], rect[2], rect[3]
            bbox = BBox(x, y, w, h)

            if _id in self.trackedObjects:
                _rect = self.trackedObjects[_id]
                _x, _y, _w, _h = _rect[0], _rect[1], _rect[2], _rect[3]
                _box = BBox(_x, _y, _w, _h)

                iouCalculator = IoUCalculator(bbox, _box)
                iou = iouCalculator.calcIoU()
                if iou < static_object_iou_threshold:
                    lst.append(_box)    # Adding only mobile objects
                    lst.append(bbox)
            # else:
            #     lst.append(bbox)
        return lst

    # def updateDetect(self, list_of_mobile_object_bboxes: list, detect: dict) -> None:
    #     """Remove the static objects (having iou >= 0.8) and gives tiles having mobile objects"""
    #     # print(list_of_mobile_object_bboxes)
    #     for bx in list_of_mobile_object_bboxes:
    #         x, y, w, h = bx.x, bx.y, bx.w, bx.h
    #         for d in detect:
    #             if (x < detect[d][0] and y < detect[d][1] and y > (detect[d][1] - detect[d][3]) and x > (detect[d][0] - detect[d][2]) and detect[d][4] == False):
    #                 detect[d][4] = True       # Select the tile which contains the top left corner of the BBox
    #             if (detect[d][0] < (x + w) and detect[d][0] > x) and (detect[d][1] > y and (detect[d][1] - detect[d][3]) < (y + h) and (d - 5) % 4 != 0): # -5 for tile indexing starting with 2
    #                 detect[d + 1][4] = True   # Select the tile in the right of current tile if BBox extends into it
    #             if (detect[d][1] < (y + h) and detect[d][1] > y) and (detect[d][0] > x and (detect[d][0] - detect[d][2]) < (x + w) and d < 14): # 14 for tile indexing starting with 2
    #                 detect[d + 4][4] = True   # Select the tile at the bottom of current tile if BBox extends into it                                      

    def updateDetect(self, list_of_mobile_object_bboxes: list, detect: dict) -> None:
        """Remove the static objects (having iou >= 0.8) and gives tiles having mobile objects"""
        # print(list_of_mobile_object_bboxes)
        totalTiles = self.tilesInOneDimension * self.tilesInOneDimension
        for bx in list_of_mobile_object_bboxes:
            x, y, w, h = bx.x, bx.y, bx.w, bx.h
            for d in detect:
                if (x < detect[d][0] and y < detect[d][1] and y > (detect[d][1] - detect[d][3]) and x > (detect[d][0] - detect[d][2]) and detect[d][4] == False):
                    detect[d][4] = True       # Select the tile which contains the top left corner of the BBox
                if (detect[d][0] < (x + w) and detect[d][0] > x) and (detect[d][1] > y and (detect[d][1] - detect[d][3]) < (y + h) and (d - (self.tilesInOneDimension + 1)) % (self.tilesInOneDimension) != 0): # Last condition skips the last coulmn of tiles
                    detect[d + 1][4] = True   # Select the tile in the right of current tile
                if (detect[d][1] < (y + h) and detect[d][1] > y) and (detect[d][0] > x and (detect[d][0] - detect[d][2]) < (x + w) and d < ((totalTiles + 1) - (self.tilesInOneDimension - 1))):                # Last condition checks if the current tile is in the last row   
                    detect[d + self.tilesInOneDimension][4] = True   # Select the tile at the bottom of current tile        

    def reset(self) -> None:
        """Clears the tracked mobile objects"""
        self.trackedObjects = {}
        self.lastSeenObjects = {}




