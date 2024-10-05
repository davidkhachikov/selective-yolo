# import torch
from ultralytics.models.yolo import YOLO
from tryalgo.union_rectangles import union_rectangles_fastest
from PIL import Image

def intersect_segments(x1, x2, x3, x4):
    if x1 >= x3:
        x1, x3 = x3, x1
        x2, x4 = x4, x2
    if x2 > x3:
        return (x3, x2)
    return None


def intersect_bboxes(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x_intersection = intersect_segments(x1, x2, x3, x4)
    y_intersection = intersect_segments(y1, y2, y3, y4)
    if x_intersection == None or y_intersection == None:
        return None
    return (x_intersection[0], y_intersection[0], x_intersection[1], y_intersection[1])
    

def intersect_many_bboxes(bboxes1, bboxes2):
    result = []
    for bbox1 in bboxes1:
        for bbox2 in bboxes2:
            curr_bbox = intersect_bboxes(bbox1, bbox2)
            if curr_bbox != None:
                result.append(curr_bbox)
    return result


def calculate_iou_diff_models(img, model1, model2):
    def get_bboxes(model, img):
        outputs = model(img)
        bboxes = []
        for output in outputs:
            for box in output.pred:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                bboxes.append((x1, y1, x2, y2))
    
    bboxes1 = get_bboxes(model1, img)
    bboxes2 = get_bboxes(model2, img)
    
    intersected_bboxes = intersect_many_bboxes(bboxes1, bboxes2)
    return union_rectangles_fastest(intersected_bboxes) / union_rectangles_fastest(bboxes1 + bboxes2)


def get_bboxes(model, img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((640, 640))
    
    outputs = model(img)
    bboxes = []
    for output in outputs:
        bboxes.extend(list(output.boxes.xyxy.detach().numpy()))
    return bboxes

def calculate_iou_diff_images(img1, img2, model):
    bboxes1 = get_bboxes(model, img1)
    bboxes2 = get_bboxes(model, img2)
    
    intersected_bboxes = intersect_many_bboxes(bboxes1, bboxes2)
    return union_rectangles_fastest(intersected_bboxes) / union_rectangles_fastest(bboxes1 + bboxes2)

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    img1 = './resources/universal_attack/karpin.jpg'
    img2 = './resources/universal_attack/adversarial_karpin.jpg'
    print(calculate_iou_diff_images(img1, img2, model))