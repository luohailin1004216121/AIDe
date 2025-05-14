import os

os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = str(2 ** 31 - 1)  # 32位有符号整数的最大值
#from pyvirtualdisplay import Display
import shutil
import numpy as np
from flask import Flask, Response, render_template, request, send_from_directory, jsonify, send_file, \
    after_this_request, url_for, redirect
import cv2
import logging
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import time
from pathlib import Path
import argparse
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import pandas as pd
import matplotlib.pyplot as plt
import threading
import queue
from datetime import datetime
from AIDetector_pytorch import Detector
import imutils
import cv2
from warning.sendwarning import sendWarn

# 在代码开头添加以下代码，设置字体为 SimHei，可支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)
#
# project_name = '/root/linalw/t5/runs'
# weight_name = '5l-5l-c1-addNew'
# weightHf_name = 'hf'
# modelDir_path = '/root/linalw/t5/weight/'
ds_host = '106.12.166.86:8090/t5/runs'
project_name = 'run'
weight_name = 'c-m-m-stratch-all'
#weight_name = 'frame6'
weightHf_name = 'c-m-m-stratch-all'
modelDir_path = 'mo'

# 存储每个任务的状态
task_states = {}

# 存储每个任务的帧队列
task_frame_queues = {}

task_thread = {}
# 新增：存储每个任务的检测器实例
task_detectors = {}

# 维护报警类别数组
alarm_categories = ['no-hat', 'smoking', 'no-belt', 'hat', 'belt']

# 存储每个任务的报警队列和报警间隔时间
task_alarm_queues = {}
task_alarm_intervals = {}
import logging

# # 获取当前时间
# current_time = datetime.now().strftime("%Y%m%d%H%M%S")
# # 生成包含时间的日志文件名
# log_filename = f'log/app_{current_time}.log'
# # 配置日志记录
# logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TaskState:
    def __init__(self):
        self.pause_flag = False
        self.stop_flag = False
        self.params = None  # 新增：存储任务参数
        self.video_feed_url = None  # 新增：存储视频流地址


def api_response(code, message, task_id=None, data=None):
    response = {
        "code": code,
        "message": message,
        "taskId": task_id,
        "data": data if data is not None else []
    }
    return jsonify(response)


@app.route('/pause', methods=['GET'])
def pause():
    try:
        # 获取 taskId 参数
        task_id = request.args.get('taskId')
        if not task_id:
            return api_response(400, "Missing 'taskId' parameter", task_id), 400
        if task_id in task_states:
            # 暂停任务
            task_states[task_id].pause_flag = True
            task_states[task_id].stop_flag = False  # 终止检测线程
            if task_id in task_thread:
                task_thread[task_id].join()  # 等待线程结束
                del task_thread[task_id]
            logging.info(f"Task {task_id} has been paused.")
            return api_response(200, "Task paused successfully", task_id), 200
        else:
            logging.warning(f"Task {task_id} not found.")
            return api_response(404, "Task not found", task_id), 404
    except Exception as e:
        logging.error(f"Error pausing task: {e}")
        return api_response(500, "Internal server error", task_id, {"error": str(e)}), 500


@app.route('/resume', methods=['GET'])
def resume():
    try:
        # 获取 taskId 参数
        task_id = request.args.get('taskId')
        if not task_id:
            return api_response(400, "Missing 'taskId' parameter", task_id), 400
        if task_id in task_states:
            if task_states[task_id].params:
                # 恢复任务
                task_states[task_id].pause_flag = False
                task_states[task_id].stop_flag = False
                # 使用保存的参数重新创建并启动检测线程
                params = task_states[task_id].params
                detector = ObjectDetector(params, task_id)
                task_detectors[task_id] = detector
                if task_id not in task_thread or not task_thread[task_id].is_alive():
                    thread = threading.Thread(target=process_task, args=(params, task_id))
                    task_thread[task_id] = thread
                    thread.start()
                logging.info(f"Task {task_id} has been resumed.")

                def generate_frames():
                    while True:
                        frame = task_frame_queues[task_id].get()
                        if frame is None:
                            break
                        yield frame

                return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
            else:
                logging.warning(f"Task {task_id} has no saved parameters.")
                return api_response(400, "Task has no saved parameters", task_id), 400
        else:
            logging.warning(f"Task {task_id} not found.")
            return api_response(404, "Task not found", task_id), 404
    except Exception as e:
        logging.error(f"Error resuming task: {e}")
        return api_response(500, "Internal server error", task_id, {"error": str(e)}), 500


@app.route('/stop', methods=['GET'])
def stop():
    try:
        # 获取 taskId 参数
        task_id = request.args.get('taskId')
        if not task_id:
            return api_response(400, "Missing 'taskId' parameter", task_id), 400
        if task_id in task_states:
            task_states[task_id].stop_flag = True
            # # 从 task_detectors 中获取对应的检测器实例
            # detector = task_detectors.get(task_id)
            # exp_name = ""
            # if detector and hasattr(detector, 'exp_name'):
            #     exp_name = detector.exp_name
            # if task_id in task_thread:
            #     task_thread[task_id].join()  # 等待线程结束
            #     del task_thread[task_id]
            return api_response(200, "Task stopped successfully", task_id), 200
        else:
            logging.warning(f"Task {task_id} not found.")
            return api_response(404, "Task not found", task_id), 404
    except Exception as e:
        logging.error(f"Error stopping task: {e}")
        return api_response(500, "Internal server error", task_id, {"error": str(e)}), 500


class ObjectDetector:
    def __init__(self, params, task_id):
        try:
            # 存储参数
            self.params = params
            # 从字典中获取参数
            self.source = self.params['source']
            self.weights = self.params['weights']
            self.view_img = self.params['view_img']
            self.save_txt = self.params['save_txt']
            self.imgsz = self.params['img_size']
            self.nosave = self.params['nosave']
            self.classes = self.params['classes']
            self.agnostic_nms = self.params['agnostic_nms']
            self.augment = self.params['augment']
            self.project = self.params['project']
            self.name = self.params['name']
            self.exist_ok = self.params['exist_ok']
            self.conf_thres = self.params['conf_thres']
            self.iou_thres = self.params['iou_thres']
            self.device = self.params['device']
            self.save_conf = self.params['save_conf']
            self.save_img = not self.nosave and not self.source.endswith('.txt')
            self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://', 'https://'))

            # Directories
            self.save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))
            (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
            self.frames_dir = self.save_dir / 'frames'
            self.frames_dir.mkdir(parents=True, exist_ok=True)

            # Initialize
            set_logging()
            if torch.cuda.is_available():
                self.device = '0'
            self.device = select_device(self.device)

            self.half = self.device.type != 'cpu'

            # Load model
            self.model = attempt_load(self.weights, map_location=self.device)
            self.stride = int(self.model.stride.max())
            self.imgsz = check_img_size(self.imgsz, s=self.stride)
            if self.half:
                self.model.half()

            # Second-stage classifier
            self.classify = False
            if self.classify:
                self.modelc = load_classifier(name='resnet101', n=2)
                self.modelc.load_state_dict(
                    torch.load('weights/resnet101.pt', map_location=self.device, weights_only=True)['model']).to(
                    self.device).eval()

            # Set Dataloader
            self.vid_path, self.vid_writer = None, None
            # if self.webcam:
            #     cudnn.benchmark = True
            #     self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
            # else:
            #     self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)

            # Get names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [(random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255)
                           for _
                           in self.names]
            self.detect_interval = params.get('detect_interval', 1)
            self.frame_count = 0
            self.task_id = task_id
            self.start_time = datetime.now().strftime("%Y%m%d%H%M%S")

            # 初始化报警队列和报警间隔时间
            task_alarm_queues[task_id] = []
            task_alarm_intervals[task_id] = params.get('alarm_interval', 3)  # 默认报警间隔为3秒
            # 启动发送报警请求的线程
            self.alarm_thread = threading.Thread(target=self.sendWarnRequest)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()
            # self.warn_thead()

            # 新增：用于存储待保存的数据
            self.images_to_save = []
            self.labels_to_save = {}
            self.video_frames = []

        except Exception as e:
            print(f"Error in ObjectDetector initialization: {e}")
            raise

    def warn_thead(self):
        if self.webcam:
            # 启动发送报警请求的线程
            self.alarm_thread = threading.Thread(target=self.sendWarnRequest)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()

    def detect_video(self):
        try:
            if self.webcam:
                cudnn.benchmark = True
                self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
            else:
                self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)

            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.model.parameters())))

            t0 = time.time()

            #path：当前帧的文件路径或视频源地址。
            #img：预处理后的图像（用于模型输入），通常是调整大小并归一化的张量
            #im0s：原始图像（用于可视化），保持原始分辨率。
            #vid_cap：视频捕获对象（仅视频流有，图像序列为 None）。
            #s：状态字符串，包含处理信息（如帧率、进度）。 （未写）
            for path, img, im0s, vid_cap in self.dataset:
                try:
                    if isinstance(path, list) and len(path) == 1:
                        path = path[0]

                    task_state = task_states.get(self.task_id)
                    if task_state and task_state.stop_flag:
                        if self.vid_writer:
                            self.vid_writer.release()
                        cv2.destroyAllWindows()
                        task_frame_queues[self.task_id].put(None)
                        self.dataset.stop()
                        break

                    if task_state and task_state.pause_flag:
                        self.dataset.stop()
                        break

                    self.frame_count += 1
                    if self.frame_count % self.detect_interval == 0:

                        # 1. 准备输入
                        img = torch.from_numpy(img).to(self.device)
                        img = img.half() if self.half else img.float()
                        img /= 255.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)

                        t1 = time_synchronized()

                        # 2. 模型推理 向前传播
                        pred = self.model(img, augment=self.augment)[0]

                        # 3. 后处理（NMS非极大值抑制）
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                                   agnostic=self.agnostic_nms)
                        t2 = time_synchronized()

                        #是 YOLOv5 目标检测框架中用于应用二次分类器的步骤，
                        # 主要用于对检测出的目标进行更精细的分类。
                        #通过二次分类过滤掉低置信度或错误的检测结果。
                        if self.classify:
                            pred = apply_classifier(pred, self.modelc, img, im0s)

                        # 4. 处理检测结果
                        if len(pred) > 0:
                            detected_classes = []
                            for i, det in enumerate(pred):
                                if self.webcam:
                                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.dataset.count
                                else:
                                    p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

                                p = Path(p)
                                save_path = str(self.save_dir / p.name)
                                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                                    '' if self.dataset.mode == 'image' else f'_{frame}')
                                s += '%gx%g' % img.shape[2:]
                                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                                if len(det):
                                    # 调整边界框坐标从模型尺寸到原始图像尺寸
                                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                                    for c in det[:, -1].unique():
                                        n = (det[:, -1] == c).sum()
                                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                                    # 绘制检测结果
                                    for *xyxy, conf, cls in reversed(det):
                                        cls_name = self.names[int(cls)]
                                        if cls_name not in detected_classes:
                                            detected_classes.append(cls_name)

                                        # 保存标签到内存
                                        if self.save_txt:
                                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                            line = (cls, *xyxy, conf) if self.save_conf else (cls, *xyxy)
                                            if txt_path not in self.labels_to_save:
                                                self.labels_to_save[txt_path] = []
                                            self.labels_to_save[txt_path].append(line)

                                        # 给图片标记标签
                                        if self.save_img or self.view_img:
                                            label = f'{self.names[int(cls)]} {conf:.2f}'
                                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)],
                                                         line_thickness=3)
                                im0s = im0
                                # 保存图片到内存
                                if self.save_img and len(det) > 0:
                                    class_names = list(set([self.names[int(cls)] for *xyxy, conf, cls in det]))
                                    class_names_str = '_'.join(class_names)
                                    timestamp = int(time.time())
                                    frame_save_path = self.frames_dir / f'{self.params["name"]}_{class_names_str}_{timestamp}.png'
                                    self.images_to_save.append((im0, frame_save_path))
                                    if self.vid_writer is None:
                                        if vid_cap:
                                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                        else:
                                            fps, w, h = 30, im0s.shape[1], im0s.shape[0]
                                        video_name = f'{Path(self.source).stem}_{self.start_time}.mp4'
                                        save_path = str(self.save_dir / video_name)
                                        self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                                          fps,
                                                                          (w, h))

                            # 检查是否有报警类别
                            for cls_name in detected_classes:
                                if cls_name in alarm_categories:
                                    current_time = time.time()
                                    alarm_queue = task_alarm_queues[self.task_id]
                                    if not alarm_queue:
                                        alarm_queue.append((current_time, cls_name))
                                    else:
                                        last_alarm_time, _ = alarm_queue[-1]
                                        if current_time - last_alarm_time > task_alarm_intervals[self.task_id]:
                                            alarm_queue.append((current_time, cls_name))

                    if isinstance(im0s, list) and len(im0s) == 1:
                        im0s = im0s[0]

                    # 保存视频帧到内存
                    self.video_frames.append(im0s)

                    #编码 100 代表了画面质量
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                    result, encoded_frame = cv2.imencode('.jpg', im0s, encode_param)
                    if not result:
                        print("Could not encode frame")
                        continue
                    frame = (b'--frame\r\n'
                             b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_frame) + b'\r\n')
                    task_frame_queues[self.task_id].put(frame)

                except Exception as e:
                    print(f"Error in processing frame: {e}")
                    continue

            # 检测完成后保存所有数据
            self.save_all_data()

            # if self.save_txt or self.save_img:
            #     s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
                # print(f"Results saved to {self.save_dir}{s}")

            # print(f'Done. ({time.time() - t0:.3f}s)')
            self.exp_name = self.name
            task_frame_queues[self.task_id].put(None)
        except Exception as e:
            task_frame_queues[self.task_id].put(None)
            print(f"Error in detect_video: {e}")

    def detect_images_or_folder(self):
        try:
            if Path(self.source).is_dir():
                image_paths = []
                image_inputs = []
                original_images = []
                image_extensions = ['.jpg', '.jpeg', '.png']
                for file in Path(self.source).iterdir():
                    if file.is_file() and file.suffix.lower() in image_extensions:
                        image_paths.append(str(file))
                        dataset = LoadImages(str(file), img_size=self.imgsz, stride=self.stride)
                        for _, img, im0s, _ in dataset:
                            image_inputs.append(img)
                            original_images.append(im0s)

                if not image_inputs:
                    return

                if self.device.type != 'cpu':
                    self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                        next(self.model.parameters())))

                image_inputs = np.stack(image_inputs, axis=0)
                img = torch.from_numpy(image_inputs).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0

                pred = self.model(img, augment=self.augment)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                           agnostic=self.agnostic_nms)

                if self.classify:
                    pred = apply_classifier(pred, self.modelc, img, original_images)

                for i, (path, im0s, det) in enumerate(zip(image_paths, original_images, pred)):
                    task_state = task_states.get(self.task_id)
                    if task_state and task_state.stop_flag:
                        break
                    while task_state and task_state.pause_flag:
                        time.sleep(1)
                        continue

                    p = Path(path)
                    save_path = str(self.save_dir / p.name)
                    txt_path = str(self.save_dir / 'labels' / p.stem) + '.txt'
                    gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            if self.save_txt:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)
                                if txt_path not in self.labels_to_save:
                                    self.labels_to_save[txt_path] = []
                                self.labels_to_save[txt_path].append(line)

                            if self.save_img or self.view_img:
                                label = f'{self.names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0s, label=label, color=self.colors[int(cls)], line_thickness=3)

                        class_names = list(set([self.names[int(cls)] for *xyxy, conf, cls in det]))
                        class_names_str = '_'.join(class_names)
                        timestamp = int(time.time())
                        frame_save_path = self.frames_dir / f'{self.params["name"]}_{class_names_str}_{timestamp}_{p.stem}.png'
                        self.images_to_save.append((im0s, frame_save_path))

            else:
                dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
                if self.device.type != 'cpu':
                    self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                        next(self.model.parameters())))

                for path, img, im0s, _ in dataset:
                    task_state = task_states.get(self.task_id)
                    if task_state and task_state.stop_flag:
                        break
                    while task_state and task_state.pause_flag:
                        while not task_frame_queues[self.task_id].empty():
                            task_frame_queues[self.task_id].get()
                        time.sleep(1)
                        continue

                    img = torch.from_numpy(img).to(self.device)
                    img = img.half() if self.half else img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = self.model(img, augment=self.augment)[0]
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                               agnostic=self.agnostic_nms)

                    if self.classify:
                        pred = apply_classifier(pred, self.modelc, img, im0s)

                    p = Path(path)
                    save_path = str(self.save_dir / p.name)
                    txt_path = str(self.save_dir / 'labels' / p.stem) + '.txt'
                    gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
                    if len(pred[0]):
                        det = pred[0]
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            if self.save_txt:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)
                                if txt_path not in self.labels_to_save:
                                    self.labels_to_save[txt_path] = []
                                self.labels_to_save[txt_path].append(line)

                            if self.save_img or self.view_img:
                                label = f'{self.names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0s, label=label, color=self.colors[int(cls)], line_thickness=3)

                        class_names = list(set([self.names[int(cls)] for *xyxy, conf, cls in det]))
                        class_names_str = '_'.join(class_names)
                        timestamp = int(time.time())
                        frame_save_path = self.frames_dir / f'{self.params["name"]}_{class_names_str}_{timestamp}_{p.stem}.png'
                        self.images_to_save.append((im0s, frame_save_path))

            # 检测完成后保存所有数据
            self.save_all_data(None)

            if self.task_id in task_states:
                task_states[self.task_id].stop_flag = True
                # 检查是否有报警类别
                # detected_classes = [self.names[int(cls)] for *xyxy, conf, cls in det]
                # for cls_name in detected_classes:
                #     if cls_name in alarm_categories:
                #         current_time = time.time()
                #         alarm_queue = task_alarm_queues[self.task_id]
                #         if not alarm_queue:
                #             alarm_queue.append((current_time, cls_name))
                #         else:
                #             last_alarm_time, _ = alarm_queue[-1]
                #             if current_time - last_alarm_time > task_alarm_intervals[self.task_id]:
                #                 alarm_queue.append((current_time, cls_name))

        except Exception as e:
            print(f"Error in detect_images_or_folder: {e}")

    def save_all_data(self):
        if self.save_img:
            # 保存图片
            for im0, frame_save_path in self.images_to_save:
                cv2.imwrite(str(frame_save_path), im0)

        if self.save_txt:
            # 保存标签
            for txt_path, lines in self.labels_to_save.items():
                with open(txt_path + '.txt', 'w') as f:
                    for line in lines:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

        # 保存视频
        if self.save_img and self.video_frames:
            for frame in self.video_frames:
                self.vid_writer.write(frame)
            self.vid_writer.release()

    def sendWarnRequest(self):

        while True:
            alarm_queue = task_alarm_queues[self.task_id]
            # print("大护法客户发扣扣",alarm_queue)
            if alarm_queue :
                timestamp, category = alarm_queue.pop(0)
                # 调用 sendWarn 方法
                if category:
                    sendWarn(self.task_id, category)
            time.sleep(task_alarm_intervals[self.task_id])
            task_state = task_states.get(self.task_id)

            if task_state is None or task_state.stop_flag:
                break


def process_task(params, task_id):
    try:
        detector = ObjectDetector(params, task_id)
        if detector.webcam or params['source'].endswith('.mp4'):  # 判断是否为视频或视频流
            detector.detect_video()
        elif Path(params['source']).is_dir():  # 判断是否为文件夹
            detector.detect_images_or_folder()
        else:  # 默认为图片
            detector.detect_images_or_folder()
    except Exception as e:
        print(f"Error in process_task: {e}")
    finally:
        task_state = task_states.get(task_id)
        if task_state and task_state.stop_flag:
            # 任务结束后，删除该任务号的相关信息
            if task_id in task_states:
                del task_states[task_id]
            if task_id in task_frame_queues:
                del task_frame_queues[task_id]
            if task_id in task_thread:
                del task_thread[task_id]
            if task_id in task_detectors:
                del task_detectors[task_id]
            if task_id in task_alarm_queues:
                del task_alarm_queues[task_id]
            if task_id in task_alarm_intervals:
                del task_alarm_intervals[task_id]


@app.route('/video_feed')
def video_feed():
    try:
        from datetime import datetime
        # 获取当前时间
        now = datetime.now()
        source = request.args.get('source')  # 从前端获取 source
        taskId = request.args.get('taskId')
        if taskId in task_states:
            return api_response(400, "Task ID already exists. Please use a unique task ID.", taskId), 400
        task_states[taskId] = TaskState()
        task_frame_queues[taskId] = queue.Queue()

        # 获取前端传入的权重模型名
        weight_model_name = Path(modelDir_path) / Path(request.args.get('weight', weight_name) + '.pt')
        # 获取前端传入的 classes 参数
        classes_str = request.args.get('classes')
        if classes_str:
            # 将逗号分隔的字符串转换为数字列表
            classes = [int(cls) for cls in classes_str.split(',')]
        else:
            classes = None

        # 新增：获取前端传入的报警间隔时间
        alarm_interval = int(request.args.get('alarm_interval', 3))

        # 定义参数的默认值
        params = {
            'weights': [weight_model_name],
            'source': source,
            'img_size': 640,
            'conf_thres': 0.35,
            'iou_thres': 0.45,
            'device': '0',
            'view_img': False,
            'save_txt': True,
            'save_conf': True,
            'nosave': False,
            'classes': classes,
            'agnostic_nms': False,
            'augment': False,
            'project': project_name,
            'name': taskId,
            'exist_ok': True,
            'alarm_interval': alarm_interval  # 新增：存储报警间隔时间
        }
        task_states[taskId].params = params
        # 创建 ObjectDetector 实例
        detector = ObjectDetector(params, taskId)
        # 将检测器实例存储到 task_detectors 中
        task_detectors[taskId] = detector

        # 创建新线程处理任务
        thread = threading.Thread(target=process_task, args=(params, taskId))
        task_thread[taskId] = thread
        thread.start()
        video_feed_url = url_for('video_feed', source=source, taskId=taskId, _external=True)
        task_states[taskId].video_feed_url = video_feed_url

        def generate_frames():
            while True:
                frame = task_frame_queues[taskId].get()
                if frame is None:
                    if taskId in task_states:
                        task_states[taskId].stop_flag = True
                    break
                yield frame

        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return api_response(500, str(e), taskId)


@app.route('/get_expList')
def get_expList():
    try:
        exp_list = []
        detect_dir = Path(project_name)
        if detect_dir.exists() and detect_dir.is_dir():
            for item in detect_dir.iterdir():
                if item.is_dir():
                    exp_list.append(item.name)
        return api_response(200, "Success", None, exp_list)
    except Exception as e:
        return api_response(500, str(e), None)


@app.route('/get_modelList')
def get_modelList():
    try:
        model_list = []
        model_dir = Path(modelDir_path)
        if model_dir.exists() and model_dir.is_dir():
            for item in model_dir.iterdir():
                if item.is_file() and item.suffix == '.pt':
                    model_list.append(item.name)
        return api_response(200, "Success", None, model_list)
    except Exception as e:
        return api_response(500, str(e), None)


@app.route('/get_weightClasses')
def get_modelClasses():
    try:
        # 从请求参数中获取权重模型文件名
        weight_filename = request.args.get('weight')
        if not weight_filename:
            return api_response(400, "Missing 'weight_filename' parameter", None), 400

        # 构建完整的权重文件路径
        weight_path = Path(modelDir_path) / f'{weight_filename}.pt'
        if not weight_path.exists():
            return api_response(404, "Model weight file not found", None), 404

        # 选择设备
        device = select_device('')
        # 加载模型
        model = attempt_load(str(weight_path), map_location=device)
        # 获取模型的类别名称
        names = model.module.names if hasattr(model, 'module') else model.names

        # 构建包含类别编号和名称的列表
        classes = [{"id": i, "name": name} for i, name in enumerate(names)]
        return api_response(200, "Success", None, classes)
    except Exception as e:
        return api_response(500, str(e), None), 500


@app.route('/download_resultsDir')
def download_resultsDir():
    try:
        results_dir = request.args.get('taskId')  # 从前端获取 results_dir
        fileType = request.args.get('fileType')
        if results_dir:
            source_dir = Path(project_name) / results_dir
            if source_dir.exists() and source_dir.is_dir():
                if fileType == '0':
                    # 当 fileType 为 0 时，获取 frames 目录下所有图片的网络地址
                    frames_dir = source_dir / 'frames'
                    if frames_dir.exists() and frames_dir.is_dir():
                        image_urls = []
                        for img_file in frames_dir.glob('*.png'):  # 假设图片都是 jpg 格式，可根据实际情况修改
                            # 生成完整的网络路径，使用 request.host 来获取当前服务器的地址和端口
                            img_file_str = str(img_file.name).replace("\\", "/")
                            image_url = f'http://{ds_host}/{results_dir}/frames/{img_file_str}'
                            image_urls.append(image_url)
                        return api_response(200, "Success", None, image_urls)
                    else:
                        return api_response(404, "Frames directory not found", None)
                elif fileType == '1':
                    # 当 fileType 为 1 时，获取 results_dir 下的视频网络地址
                    video_files = list(source_dir.glob('*.mp4'))  # 假设视频都是 mp4 格式，可根据实际情况修改
                    if video_files:
                        video_urls = []
                        for video_file in video_files:
                            # 生成完整的网络路径，使用 request.host 来获取当前服务器的地址和端口
                            video_file_str = str(video_file.name).replace("\\", "/")
                            video_url = f'http://{ds_host}/{results_dir}/{video_file_str}'
                            video_urls.append(video_url)
                        return api_response(200, "Success", None, video_urls)
                    else:
                        return api_response(404, "No video files found", None)
                else:
                    # 如果 fileType 不是 0 或 1，直接提示参数错误
                    return api_response(400, "Invalid fileType parameter", None)
            else:
                return api_response(404, "Directory not found", None)
        else:
            return api_response(400, "Invalid request", None)
    except Exception as e:
        return api_response(500, str(e), None)


@app.route('/image_detection')
def image_detection():
    try:
        source = request.args.get('source')  # 从前端获取 source
        if not source:
            return api_response(400, "Missing 'source' parameter", None), 400

        taskId = request.args.get('taskId')  # 生成唯一的 taskId
        if taskId in task_states:
            return api_response(400, "Task ID already exists. Please use a unique task ID.", taskId), 400
        task_states[taskId] = TaskState()
        task_frame_queues[taskId] = queue.Queue()

        # 获取前端传入的权重模型名
        weight_model_name = Path(modelDir_path) / Path(request.args.get('weight', weightHf_name) + '.pt')
        print(weight_model_name)
        # 获取前端传入的 classes 参数
        classes_str = request.args.get('classes')
        if classes_str:
            # 将逗号分隔的字符串转换为数字列表
            classes = [int(cls) for cls in classes_str.split(',')]
        else:
            classes = None

        # 新增：获取前端传入的报警间隔时间
        alarm_interval = int(request.args.get('alarm_interval', 3))

        # 定义参数的默认值
        params = {
            'weights': [weight_model_name],
            'source': source,
            'img_size': 640,
            'conf_thres': 0.3,
            'iou_thres': 0.45,
            'device': '0',
            'view_img': False,
            'save_txt': True,
            'save_conf': True,
            'nosave': False,
            'classes': classes,
            'agnostic_nms': False,
            'augment': False,
            'project': project_name,
            'name': taskId,
            'exist_ok': True,
            'alarm_interval': alarm_interval  # 新增：存储报警间隔时间
        }

        detector = ObjectDetector(params, taskId)
        task_detectors[taskId] = detector
        # 创建新线程处理任务
        thread = threading.Thread(target=process_task, args=(params, taskId))
        task_thread[taskId] = thread
        thread.start()
        # save_dir = Path(increment_path(Path(project_name) / taskId))
        save_dir = Path(project_name)
        frames_dir = save_dir / str(taskId) / 'frames'

        # print("frames_dir",frames_dir)
        # print("taskid",taskId)
        # print("save_dir",save_dir)
        # time.sleep(8)
        image_urls = []
        if frames_dir.exists() and frames_dir.is_dir():
            for img_file in frames_dir.glob('*.png'):
                image_url_str = str(img_file.name).replace("\\", "/")
                image_url = f'http://{ds_host}/{taskId}/frames/{image_url_str}'
                image_urls.append(image_url)
        print(image_urls)

        return api_response(200, "Success", taskId, image_urls)
    except Exception as e:
        return api_response(500, str(e), None)





class VideoCounter:
    def __init__(self, video_path, task_id, project_name='run', conf_thres=0.3, iou_thres=0.45, device='0'):
        self.video_path = video_path
        self.task_id = task_id
        self.project_name = project_name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

        # 初始化保存路径
        self.save_dir = Path(increment_path(Path(self.project_name) / self.task_id))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.video_basename = Path(video_path).stem
        self.timestamp = int(time.time())
        self.video_output_path = self.save_dir / f"{self.video_basename}_{self.timestamp}_result.mp4"
        self.txt_output_path = self.save_dir / f"{self.video_basename}_{self.timestamp}.txt"

        # 初始化检测器
        self.detector = Detector()
        self.cap = None
        self.video_writer = None
        self.object_count = {}
        self.tracked_ids = set()
        self.frames = []

    def process(self):
        """处理视频并返回计数结果"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError("Failed to open video file")

            # 逐帧处理

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                result = self.detector.feedCap(frame)

                processed_frame = result['frame']

                bboxes2draw = result['bboxes2draw']

                # 收集跟踪ID进行计数（假设bboxes2draw格式为[x1,y1,x2,y2,cls,track_id]）
                for *_, cls, track_id in bboxes2draw:
                    if track_id not in self.tracked_ids:
                        self.tracked_ids.add(track_id)
                        self.object_count[cls] = self.object_count.get(cls, 0) + 1

                self.frames.append(processed_frame)  # 暂存处理后的帧


            # 处理完成后保存视频（统一写入，避免逐帧写入的性能问题）
            self._save_video()
            self._save_count_result()

            return self.object_count

        except Exception as e:
            print(f"Video processing error: {str(e)}")
            raise
        finally:
            self._release_resources()

    def _save_video(self):

        """保存处理后的视频"""
        if not self.frames:
            return

        # 获取第一帧尺寸
        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(self.video_output_path), fourcc,
                                            self.cap.get(cv2.CAP_PROP_FPS), (width, height))

        for frame in self.frames:
            self.video_writer.write(frame)

    def _save_count_result(self):
        """保存计数结果到TXT"""
        with open(self.txt_output_path, 'w') as f:
            for cls, count in self.object_count.items():
                f.write(f"{cls}: {count}\n")

    def _release_resources(self):
        """释放资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()


# 修改/videoCount路由处理函数
@app.route('/videoCount')
def video_count():
    video_path = request.args.get('source')
    task_id = request.args.get('taskId')

    if not video_path or not task_id:
        return api_response(400, "Missing required parameters (source or taskId)", task_id), 400

    try:
        # 初始化视频计数器
        counter = VideoCounter(
            video_path=video_path,
            task_id=task_id,
            project_name=project_name,
        )

        # 执行计数处理

        count_result = counter.process()

        return api_response(200, "Processing completed", task_id, count_result)

    except Exception as e:
        return api_response(500, f"Video processing failed: {str(e)}", task_id), 500


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return api_response(500, str(e), None)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
