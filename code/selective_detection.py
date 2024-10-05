import os
from ultralytics import YOLO
from torch import device, cuda, load
from iou import calculate_iou_diff_models_batch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import fiftyone


device_torch = device("cuda" if cuda.is_available() else "cpu")

def get_weights(n):
    """
    Property of weights -- all of them in range [0, 1].
    """
    k = 0.05
    return [1 - (i / (n - 1))**k for i in range(0, n)]

def load_yolo_checkpoints(n=20, step=5, folder_path=".\\checkpoints", prefix="yolo_AdamW_checkpoint", yolo_version="yolov8n"):
    loaded_checkpoints = []
    
    for i in range(0, n*step, step):
        checkpoint_name = f"{prefix}_{i}.pt"
        checkpoint_path = os.path.join(folder_path, checkpoint_name)
        
        print(f"Loading checkpoint: {checkpoint_path}")
        loaded_checkpoints.append(
        YOLO(yolo_version + '.yaml').load_state_dict(
                load(
                    checkpoint_path,
                    map_location=device_torch
                )['model_state_dict']
            )
        )
    
    return loaded_checkpoints


def votinig(dataloader, checkpoints):
    weights = get_weights(len(checkpoints))
    num_checkpoints = len(checkpoints)
    num_images = len(dataloader)
    
    divergences = np.zeros((num_checkpoints - 1, num_images))
    
    last_checkpoint = checkpoints[-1]
    
    for i in range(num_checkpoints - 1):
        checkpoint = checkpoints[i]
        
        for j, batch in enumerate(dataloader):
            divergences[i, j:j+len(batch)] = calculate_iou_diff_models_batch(batch, last_checkpoint, checkpoint)
    
    return np.dot(weights[:-1], divergences)


def votinig_parallelized_experimental(dataloader, checkpoints):
    weights = get_weights(len(checkpoints))
    num_checkpoints = len(checkpoints)
    num_images = len(dataloader)
    
    divergences = np.zeros((num_checkpoints - 1, num_images))
    last_checkpoint = checkpoints[-1]
    
    def process_checkpoint(i):
        checkpoint = checkpoints[i]
        divergence = np.zeros(num_images)
        for j, batch in enumerate(dataloader):
            divergence[j:j+len(batch)] = calculate_iou_diff_models_batch(batch, last_checkpoint, checkpoint)
        return i, divergence
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_checkpoint, i) for i in range(num_checkpoints - 1)]
        
        for future in futures:
            i, divergence = future.result()
            divergences[i] = divergence
    
    return np.dot(weights[:-1], divergences)


def get_coco_dataloader(batch_size=4, shuffle=True, num_workers=4):
    coco_dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")

    coco_dataloader = DataLoader(dataset=coco_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return coco_dataloader
