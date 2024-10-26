import os
from ultralytics import YOLO
from torch import device, cuda, load
from iou import calculate_iou_diff_models_batch
from multiprocessing import Manager, shared_memory, Pool, Array
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import DataLoader, Dataset
import fiftyone.zoo as foz
from ctypes import c_double
import json

device_torch = device("cuda" if cuda.is_available() else "cpu")

def get_weights(n):
    """
    Property of weights -- all of them in range [0, 1].
    """
    k = 0.5
    w = np.array([(i / n)**k for i in range(1, n+1)])
    return w / w.sum()

def load_yolo_checkpoints(n=20, step=5, folder_path=".\\checkpoints", prefix="epoch", yolo_version="yolo11n"):
    loaded_checkpoints = []
    
    for i in range(0, n*step, step):
        checkpoint_name = f"{prefix}{i}.pt"
        checkpoint_path = os.path.join(folder_path, checkpoint_name)
        
        print(f"Loading checkpoint: {checkpoint_path}")
        loaded_checkpoints.append(YOLO(checkpoint_path))
    
    return loaded_checkpoints

def voting(dataloader, checkpoints):
    weights = get_weights(len(checkpoints)-1)
    num_checkpoints = len(checkpoints)
    num_images = len(dataloader.dataset)
    
    divergences = np.zeros((num_checkpoints - 1, num_images))
    
    last_checkpoint = checkpoints[-1]
    
    for i in range(num_checkpoints - 1):
        checkpoint = checkpoints[i]
        
        for j, batch in enumerate(dataloader):
            iou = calculate_iou_diff_models_batch(batch, last_checkpoint, checkpoint)
            print(iou)
            divergences[i, j*len(batch):j*len(batch)+len(batch)] = iou
    return divergences, weights @ divergences

def process_checkpoint_shared(i, checkpoint, last_checkpoint, dataloader, shared_divergences):
    def process_checkpoint(i, checkpoint, last_checkpoint, dataloader, verbose=True):
        divergence = []
        for batch in dataloader:
            iou = calculate_iou_diff_models_batch(batch, last_checkpoint, checkpoint)
            divergence.append(iou)
            if verbose:
                print(f"{i} vs last: {iou}")
        return np.concatenate(divergence)
    divergence = process_checkpoint(i, checkpoint, last_checkpoint, dataloader)
    shared_divergences[i, :] = divergence
    return i

def voting_parallelized_experimental(dataloader, checkpoints):
    # Get weights and necessary dimensions
    weights = get_weights(len(checkpoints))
    num_checkpoints = len(checkpoints)
    num_images = len(dataloader.dataset)
    
    # Create a shared Array for storing divergences as a flat list (row-major order)
    shared_divergences = Array(c_double, (num_checkpoints - 1) * num_images)
    
    # Manager lists to share read-only data among processes
    manager = Manager()
    shared_checkpoints = manager.list(checkpoints)
    shared_dataloader = manager.list(dataloader)
    
    # Define arguments for each checkpoint's processing
    args = [
        (i, shared_checkpoints[i], shared_checkpoints[-1], shared_dataloader, shared_divergences, num_images) 
        for i in range(num_checkpoints - 1)
    ]

    # Use multiprocessing.Pool to distribute the computation
    with Pool() as pool:
        pool.starmap(process_checkpoint_shared, args)

    # Reshape shared_divergences array after pool processing
    final_divergences = np.frombuffer(shared_divergences.get_obj()).reshape((num_checkpoints - 1, num_images))

    # Calculate weighted sum of divergences
    weighted_divergences = np.dot(weights[:-1], final_divergences)

    return final_divergences, weighted_divergences

class DatasetOfPaths(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.filepaths[idx]

def get_coco_dataloader(n_images=64, batch_size=8):
    # Load the COCO dataset
    coco_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="test",
        max_samples=int(n_images),
        shuffle=False,
    )

    # Convert to a PyTorch-compatible dataset
    pytorch_dataset = DatasetOfPaths([sample.filepath for sample in coco_dataset])

    # Create the DataLoader
    coco_dataloader = DataLoader(dataset=pytorch_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return coco_dataloader

if __name__ == '__main__':
    persentage_of_lowest = 0.2
    n_images = 512
    n_checkpoints = 20

    checkpoints = load_yolo_checkpoints(n_checkpoints, folder_path='./data/checkpoints/AdamW/')
    dataloader = get_coco_dataloader(n_images)
    
    divergences, eval = voting(dataloader, checkpoints)
    indices_of_lowest_n = np.argsort(eval)[:int(persentage_of_lowest*n_images)]
    indices_of_best_n = np.argsort(eval)[int((1-persentage_of_lowest)*n_images):]

    print("\n\n\nThe top 20 most inaccurate:")
    print([dataloader.dataset[i] for i in indices_of_lowest_n])
    print("\n\n\nThe top 20% most accurate:")
    print([dataloader.dataset[i] for i in indices_of_best_n])

    np.save('divergences.npy', divergences)
    np.save('eval.npy', eval)
    with open('divergences.json', 'w') as f:
        json.dump(divergences.tolist(), f)
    with open('eval.json', 'w') as f:
        json.dump(eval.tolist(), f)
