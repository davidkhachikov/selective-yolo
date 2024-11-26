import torch
from ultralytics import YOLO
from tryalgo.union_rectangles import union_rectangles_fastest
from PIL import Image
import numpy as np

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Функция для подсчета Precision и Recall
def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    true_positive = 0
    false_positive = 0
    false_negative = len(gt_boxes)
    
    for pred_box in pred_boxes:
        matched = False
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box[:4], gt_box[:4])
            if iou >= iou_threshold:
                true_positive += 1
                false_negative -= 1
                matched = True
                break
        if not matched:
            false_positive += 1

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    return precision, recall

def calculate_mean_iou(bboxes1, bboxes2, wrong_label_penalty=0.5, iou_threshold=0.5):
    iou_scores = []
    for box1 in bboxes1:
        for box2 in bboxes2:
            iou = calculate_iou(box1[:4], box2[:4])
            if iou > iou_threshold:  # Сравниваем только после прохождения порога
                # Если метка класса не совпадает, то не беда -- оштрафуем немножко и добавим.
                iou_scores.append(iou if box1[5] == box2[5] else wrong_label_penalty * iou)

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return mean_iou

def get_stat_bboxes_batch(imgs, model):
    results = model(imgs, verbose=False)
    ret = []
    for result in results:
        pred_boxes = []
        # print(result[0])
        for box, conf, cls in zip(
            result.boxes.xyxy.cpu().numpy(),   # Bounding box coordinates for all detections
            result.boxes.conf.cpu().numpy(),   # Confidence scores for all detections
            result.boxes.cls.cpu().numpy()     # Class labels for all detections
        ):
            x1, y1, x2, y2 = box   # Unpack the bounding box coordinates
            pred_boxes.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
        ret.append(pred_boxes)
    return ret


def calculate_stat_models_batch(imgs, model1, model2, wrong_label_penalty=0.2, iou_threshold=0.2, max_length=10, padding_value=0):
    def pad_or_truncate(lst, length, padding_value):
        """Ensure a list is of the specified length by truncating or padding with a placeholder."""
        return (lst[:length] + [padding_value] * max(0, length - len(lst)))

    stats = []
    results1 = get_stat_bboxes_batch(imgs, model1)
    results2 = get_stat_bboxes_batch(imgs, model2)

    for pred_boxes1, pred_boxes2 in zip(results1, results2):
        mean_iou = calculate_mean_iou(pred_boxes1, pred_boxes2, wrong_label_penalty, iou_threshold)
        precision, recall = calculate_precision_recall(pred_boxes1, pred_boxes2)
                
        to_append = [mean_iou, precision, recall]
        stats.append(to_append)
    
    return stats


def calculate_stat_model_batch(imgs, model):
    stats = []
    results = get_stat_bboxes_batch(imgs, model)

    for pred_boxes in results:
        # print(pred_boxes)
        if pred_boxes == []:
            stats.append([0, 0, 0, 0, 0])
            continue
        num_pred = len(pred_boxes)
        num_unique = len(np.unique([item[4] for item in pred_boxes]))
        avg_conf = np.mean([item[4] for item in pred_boxes])

        area = union_rectangles_fastest([item[:4] for item in pred_boxes])

        center = np.mean([item[:4] for item in pred_boxes])

        to_append = [avg_conf, num_pred, num_unique / num_pred, area, center]
        stats.append(to_append)
    
    return stats