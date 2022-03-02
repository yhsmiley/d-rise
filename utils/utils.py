import math

import cv2
import numpy as np
from numpy.linalg import norm


def generate_mask(image_size, grid_size, prob_thresh, rng):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.floor(image_w / grid_w), math.floor(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (rng.uniform(0, 1, size=(grid_h, grid_w)) < prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = rng.integers(0, cell_w, endpoint=True)
    offset_h = rng.integers(0, cell_h, endpoint=True)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask

def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) * 255).astype(np.uint8)
    return masked

def _iou_batch(bbox1, bboxes2):
    bbox1 = np.asarray(bbox1)
    bboxes2 = np.asarray(bboxes2)
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def _cosine_similarity(arr1, arr2):
    return np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))

def similarity_metric(target_box, target_one_hot, detections):
    all_bboxes = detections[:, :4]
    all_objectness = detections[:, 4]
    all_cls_probs = detections[:, 5:]

    ious = _iou_batch(target_box, all_bboxes).squeeze()
    cosines = _cosine_similarity(all_cls_probs, target_one_hot)

    return np.max(ious * np.asarray(all_objectness) * cosines)
