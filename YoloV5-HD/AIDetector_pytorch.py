# import torch
# import numpy as np
# from models.experimental import attempt_load
# from utils.general import non_max_suppression, scale_coords
# from utils.BaseDetector import baseDet
# from utils.torch_utils import select_device
# from utils.datasets import letterbox
#
# class Detector(baseDet):
#
#     def __init__(self):
#         super(Detector, self).__init__()
#         self.init_model()
#         self.build_config()
#
#     def init_model(self):
#
#         self.weights = r'mo/c-m-m-stratch-all.pt'
#         self.device = '0' if torch.cuda.is_available() else 'cpu'
#         self.device = select_device(self.device)
#         model = attempt_load(self.weights, map_location=self.device)
#         model.to(self.device).eval()
#         model.half()
#         # torch.save(model, 'test.pt')
#         self.m = model
#         self.names = model.module.names if hasattr(
#             model, 'module') else model.names
#
#     def preprocess(self, img):
#
#         img0 = img.copy()
#         img = letterbox(img, new_shape=self.img_size)[0]
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img)
#         img = torch.from_numpy(img).to(self.device)
#         img = img.half()  # 半精度
#         img /= 255.0  # 图像归一化
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         return img0, img
#
#     def detect(self, im):
#
#         im0, img = self.preprocess(im)
#
#         pred = self.m(img, augment=False)[0]
#         pred = pred.float()
#         pred = non_max_suppression(pred, self.threshold, 0.4)
#
#         pred_boxes = []
#         for det in pred:
#
#             if det is not None and len(det):
#                 det[:, :4] = scale_coords(
#                     img.shape[2:], det[:, :4], im0.shape).round()
#
#                 for *x, conf, cls_id in det:
#                     lbl = self.names[int(cls_id)]
#                     # if not lbl in ['person', 'car', 'truck']:
#                     #     continue
#                     x1, y1 = int(x[0]), int(x[1])
#                     x2, y2 = int(x[2]), int(x[3])
#                     pred_boxes.append(
#                         (x1, y1, x2, y2, lbl, conf))
#
#         return im, pred_boxes
import logging

import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.BaseDetector import baseDet
from utils.torch_utils import select_device
from utils.datasets import letterbox
from tracker import update_tracker
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

    def init_model(self):
        # self.weights = r'mo/c-m-m-stratch-all.pt'
        self.weights = r'mo/frame.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        self.m = model
        self.names = model.module.names if hasattr(model, 'module') else model.names

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):
        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        # pred = non_max_suppression(pred, self.threshold, 0.2)

        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=self.threshold)  # NMS 操作，可调整 iou_thres

        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes

    def feedCap(self, im):
        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': [],
            'bboxes2draw': []
        }
        self.frameCounter += 1
        # 记录日志
        logging.info(f'frameCounter,{self.frameCounter}')
        logging.info(f'idnum,{self.deepsort.tracker._next_id}')

        im, faces, face_bboxes, bboxes2draw = update_tracker(self, im,self.deepsort)

        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes
        retDict['bboxes2draw'] = bboxes2draw

        return retDict
