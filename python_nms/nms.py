import numpy as np
from typing import Tuple
from torchvision.ops import nms as torch_nms
import torch


def nms_torch(boxes, scores, threshold):
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(scores)
    keep_idx = torch_nms(boxes, scores, threshold)
    return keep_idx.numpy()


# modified from https://github.com/zipengbo/keras-mtcnn/blob/master/tools_matrix.py
def nms_method1(rectangles, scores, threshold):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = scores
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    return pick


# copied from https://blog.csdn.net/weixin_40920290/article/details/90602706
def nms_method2(bounding_boxes, confidences, threshold):
    """
    Args:
        bounding_boxes: np.array([(x1, y1, x2, y2), ...])
        confidences: np.array(conf1, conf2, ...),数量需要与bounding box一致,并且一一对应
        threshold: IOU阀值,若两个bounding box的交并比大于该值，则置信度较小的box将会被抑制

    Returns:
        bounding_boxes: 经过NMS后的bounding boxes
        confidences: 经过NMS后的confidences
    """
    len_bound = bounding_boxes.shape[0]
    len_conf = confidences.shape[0]
    if len_bound != len_conf:
        raise ValueError("Bounding box 与 Confidence 的数量不一致")
    if len_bound == 0:
        return np.array([]), np.array([])
    bounding_boxes, confidences = bounding_boxes.astype(np.float), np.array(confidences)

    x1, y1, x2, y2 = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)

    pick = []
    while len(idxs) > 0:
        # 因为idxs是从小到大排列的，last_idx相当于idxs最后一个位置的索引
        last_idx = len(idxs) - 1
        # 取出最大值在数组上的索引
        max_value_idx = idxs[last_idx]
        # 将这个添加到相应索引上
        pick.append(max_value_idx)

        xx1 = np.maximum(x1[max_value_idx], x1[idxs[: last_idx]])
        yy1 = np.maximum(y1[max_value_idx], y1[idxs[: last_idx]])
        xx2 = np.minimum(x2[max_value_idx], x2[idxs[: last_idx]])
        yy2 = np.minimum(y2[max_value_idx], y2[idxs[: last_idx]])

        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)

        iou = w * h / (areas[idxs[: last_idx]] + 1e-6)
        # 删除最大的value,并且删除iou > threshold的bounding boxes
        idxs = np.delete(idxs, np.concatenate(([last_idx], np.where(iou > threshold)[0])))

    # bounding box 返回一定要int类型,否则Opencv无法绘制
    return pick
