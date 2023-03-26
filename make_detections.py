import time
import cv2
import json
import numpy as np
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.apis import inference_detector


VIDEO = 'data/videos/video0015_cut.mp4'
CONFIG = 'data/models/cascade_rcnn_r50_fpn_1x.py'
CHECKPOINT = 'data/models/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth'

THRESHOLD = 0.2
RESULT = 'detections.json'


times = []  # detection times
detections = {}
model = init_detector(CONFIG, CHECKPOINT, device='cuda:0')


cap = cv2.VideoCapture(VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'num_frames: {num_frames}, fps: {fps}', '\nsize:', (width, height), '\n')


frame_id = 0
for i in tqdm(range(num_frames)):
    _, frame = cap.read()

    # make detection every second
    if i % fps != 0:
        continue

    start = time.perf_counter()
    result = inference_detector(model, frame)
    times.append(time.perf_counter() - start)

    # only person class
    boxes = result[0]

    scores = boxes[:, 4]
    boxes = boxes[scores > THRESHOLD]

    scores = boxes[:, 4]
    boxes = boxes[:, :4]
    boxes = boxes.astype(np.int32)

    xmin, ymin, xmax, ymax = np.split(boxes, 4, axis=1)
    xmin = np.clip(xmin, 0, width)
    ymin = np.clip(ymin, 0, height)
    xmax = np.clip(xmax, 0, width)
    ymax = np.clip(ymax, 0, height)
    boxes = np.concatenate([xmin, ymin, xmax, ymax], axis=1)

    assert (xmin < xmax).all()
    assert (ymin < ymax).all()

    boxes = boxes.tolist()
    scores = [round(s, 3) for s in scores.tolist()]
    detections[i] = {'boxes': boxes, 'scores': scores}

cap.release()
times = np.array(times[10:])
print('inference time:')
print(f'mean {times.mean():.3f}, std {times.std():.3f}')


with open(RESULT, 'w') as f:
    json.dump(detections, f)
