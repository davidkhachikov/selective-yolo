import json
import os
import yaml

def _coco2yolo(coco_annotation_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)

    def convert_bbox(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    for image in coco_data['images']:
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']

        yolo_annotation_file = os.path.join(output_dir, f"{image['file_name'].split('.')[0]}.txt")
        
        with open(yolo_annotation_file, 'w') as yolo_f:
            for ann in coco_data['annotations']:
                if ann['image_id'] == image_id:
                    class_id = ann['category_id'] - 1
                    bbox = ann['bbox']  # [x_min, y_min, width, height]
                    yolo_bbox = convert_bbox((image_width, image_height), bbox)
                    
                    yolo_f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

def write_dataset_yaml(coco_annotation_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(coco_annotation_path, 'r') as f:
        coco_labels = json.load(f)['categories']
    data = {"path": "", "train": "images/train", "val": "images/val", "test": "", "names": {}}
    for item in coco_labels:
        data['names'][item['id']-1] = item['name']
    with open(os.path.join(output_dir, 'coco.yml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def coco2yolo(data_path='./data/raw/coco2017/', output_path='./data/raw/coco2017/'):
    # _coco2yolo(os.path.join(data_path, './train/labels.json'), os.path.join(output_path, './train/train'))
    # _coco2yolo(os.path.join(data_path, './val/labels.json'), os.path.join(output_path, './val/val'))
    write_dataset_yaml(os.path.join(data_path, './train/labels.json'), output_path)