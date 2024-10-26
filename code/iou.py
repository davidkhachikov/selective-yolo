from ultralytics.models.yolo import YOLO
from tryalgo.union_rectangles import union_rectangles_fastest
from PIL import Image

def intersect_segments(x1, x2, x3, x4):
    """
    Calculate the intersection of two line segments [x1, x2] and [x3, x4].
    Return the intersection coordinates if they exist, otherwise None.
    """
    assert x1 <= x2 and x3 <= x4
    if x1 >= x3:
        x1, x3 = x3, x1
        x2, x4 = x4, x2
    if x2 >= x3:
        return (x3, x2)
    return None


def intersect_bboxes(bbox1, bbox2):
    """
    Calculate the intersection of two rectangular bounding boxes.
    Return the intersection coordinates if they exist, otherwise None.
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x_intersection = intersect_segments(x1, x2, x3, x4)
    y_intersection = intersect_segments(y1, y2, y3, y4)
    if x_intersection == None or y_intersection == None:
        return None
    return (x_intersection[0], y_intersection[0], x_intersection[1], y_intersection[1])
    

def intersect_many_bboxes(bboxes1, bboxes2):
    """
    Find all intersections between two sets of rectangular bounding boxes.
    This intersections may overlap!
    Return a list of intersection coordinates.
    """
    result = []
    for i, bbox1 in enumerate(bboxes1):
        for bbox2 in bboxes2[i:]:
            curr_bbox = intersect_bboxes(bbox1, bbox2)
            if curr_bbox != None:
                result.append(curr_bbox)
    return result


def get_bboxes(model, img):
    """
    Extract bounding boxes from the model's prediction on the given image.
    Resize the image to 640x640 pixels before processing.
    """
    img = img.convert('RGB')
    img = img.resize((640, 640))
    
    outputs = model(img)
    bboxes = []
    for output in outputs:
        bboxes.extend(list(output.boxes.xyxy.detach().numpy()))
    return bboxes


def calculate_iou_diff_models(img, model1, model2):
    """
    Calculate the Intersection over Union (IoU) difference between two models
    on the same image. This metric compares overlapping areas between predictions.
    """
    bboxes1 = get_bboxes(model1, img)
    bboxes2 = get_bboxes(model2, img)
    
    intersected_bboxes = intersect_many_bboxes(bboxes1, bboxes2)
    return union_rectangles_fastest(intersected_bboxes) / union_rectangles_fastest(bboxes1 + bboxes2)


def calculate_iou_diff_images(img1, img2, model):
    """
    Calculate the Intersection over Union (IoU) difference between two images
    processed by the same model. This metric compares overlapping areas
    between predictions on different input images.
    """
    bboxes1 = get_bboxes(model, img1)
    bboxes2 = get_bboxes(model, img2)
    
    intersected_bboxes = intersect_many_bboxes(bboxes1, bboxes2)
    return union_rectangles_fastest(intersected_bboxes) / union_rectangles_fastest(bboxes1 + bboxes2)


def get_bboxes_batch(model, imgs):
    """
    Extract bounding boxes from the model's predictions on a batch of images.
    The model should handle batch processing. Returns a list of bounding boxes 
    for each image in the batch.
    """
    assert isinstance(imgs, list)
    if all(isinstance(img, str) for img in imgs):
        imgs = [Image.open(img).convert('RGB').resize((640, 640)) for img in imgs]
    else:
        imgs = [img.convert('RGB').resize((640, 640)) for img in imgs]
    
    outputs = model(imgs, verbose=False)
    
    bboxes_batch = []
    for output in outputs:
        bboxes = list(output.boxes.xyxy.detach().numpy())
        bboxes_batch.append(bboxes)
    
    return bboxes_batch


def intersect_many_bboxes_batch(bboxes_batch1, bboxes_batch2):
    """
    Find all intersections between two batches of bounding boxes.
    Returns a list of intersection bounding boxes for each corresponding pair
    of bounding box batches.
    """
    result_batch = []
    for bboxes1, bboxes2 in zip(bboxes_batch1, bboxes_batch2):
        result = intersect_many_bboxes(bboxes1, bboxes2)
        result_batch.append(result)
    
    return result_batch


def calculate_iou_diff_models_batch(imgs, model1, model2):
    """
    Calculate the IoU difference between two models over a batch of images.
    This metric compares overlapping areas between predictions.
    """
    bboxes_batch1 = get_bboxes_batch(model1, imgs)
    bboxes_batch2 = get_bboxes_batch(model2, imgs)
    
    intersected_bboxes_batch = intersect_many_bboxes_batch(bboxes_batch1, bboxes_batch2)
    
    iou_diffs = []
    for intersected_bboxes, bboxes1, bboxes2 in zip(intersected_bboxes_batch, bboxes_batch1, bboxes_batch2):
        union_area = union_rectangles_fastest(bboxes1 + bboxes2)
        intersect_area = union_rectangles_fastest(intersected_bboxes)
        iou_diff = float(min(intersect_area / union_area, 1.0)) if union_area > 1e-3 else 1.0
        iou_diffs.append(iou_diff)
    
    return iou_diffs


def calculate_iou_diff_images_batch(imgs1, imgs2, model):
    """
    Calculate the IoU difference between two batches of images processed by the same model.
    This metric compares overlapping areas between predictions on different input images.
    """
    bboxes_batch1 = get_bboxes_batch(model, imgs1)
    bboxes_batch2 = get_bboxes_batch(model, imgs2)
    
    intersected_bboxes_batch = intersect_many_bboxes_batch(bboxes_batch1, bboxes_batch2)
    
    iou_diffs = []
    for intersected_bboxes, bboxes1, bboxes2 in zip(intersected_bboxes_batch, bboxes_batch1, bboxes_batch2):
        union_area = union_rectangles_fastest(bboxes1 + bboxes2)
        intersect_area = union_rectangles_fastest(intersected_bboxes)
        iou_diff = float(intersect_area / union_area)
        iou_diffs.append(iou_diff)
    
    return iou_diffs


if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    img1 = Image.open('./resources/universal_attack/karpin.jpg')
    img2 = Image.open('./resources/universal_attack/adversarial_karpin.jpg')
    print(calculate_iou_diff_images_batch([img1, img2, img2], [img2, img1, img2], model))