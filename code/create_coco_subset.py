import fiftyone.zoo as foz
import fiftyone as fo
import os
import json
from coco2yolo import coco2yolo

def get_coco_n(n: int = 12500, val_size: float = 0.2, export_dir='./data/raw/coco2017'):
    # Load validation samples
    val_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        max_samples=int(n*val_size),
        shuffle=True,
    )

    val_dataset.export(
        export_dir=os.path.join(export_dir + '/val'),
        dataset_type=fo.types.COCODetectionDataset,
    )

    # Load training samples
    train_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        max_samples=int(n*(1-val_size)),
        shuffle=True,
    )

    train_dataset.export(
        export_dir=os.path.join(export_dir + '/train'),
        dataset_type=fo.types.COCODetectionDataset,
    )

    # Print some information about the datasets
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training samples: {len(train_dataset)}")

def modify_json(json_path):
    # Open and read the JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Update json_data with the content of the 'info' key and then delete 'info'
    if "info" in json_data:
        json_data.update(json_data['info'])
        del json_data['info']
        
    # Write the modified JSON data back to the file
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def modify_jsons(jsons_path='./data/raw/coco2017'):
    # Correct path concatenation (avoid leading slashes)
    modify_json(os.path.join(jsons_path, 'val', 'labels.json'))
    modify_json(os.path.join(jsons_path, 'train', 'labels.json'))


if '__main__' == __name__:
    # get_coco_n()
    coco2yolo()