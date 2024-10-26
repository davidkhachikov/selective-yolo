import os
from ultralytics import YOLO
from utils import calculate_iou_diff_models_batch
from multiprocessing import Manager, Pool, Array
import numpy as np
from torch.utils.data import DataLoader, Dataset
import fiftyone.zoo as foz
from ctypes import c_double
import json

class EnsembleOfCheckpoints:
    def __init__(self, n_checkpoints=20, step=5, folder_path=".\\checkpoints", prefix="epoch"):
        self.n_checkpoints = n_checkpoints
        self.step = step
        self.folder_path = folder_path
        self.prefix = prefix
        self.checkpoints = self.load_checkpoints()
        self.weights = self.get_weights(len(self.checkpoints) - 1)
    
    def get_weights(self, n, k = 0.5):
        """
        Generates weights for checkpoints such that their sum equals to 1, with a "smoothnest" parameter k.
        """
        w = np.array([(i / n)**k for i in range(1, n + 1)])
        return w / w.sum()

    def load_checkpoints(self):
        """
        Loads YOLO checkpoints based on initialized parameters.
        """
        loaded_checkpoints = []
        
        for i in range(0, self.n_checkpoints * self.step, self.step):
            checkpoint_name = f"{self.prefix}{i}.pt"
            checkpoint_path = os.path.join(self.folder_path, checkpoint_name)
            
            print(f"Loading checkpoint: {checkpoint_path}")
            loaded_checkpoints.append(YOLO(checkpoint_path))
        
        return loaded_checkpoints
    
    def vote_one_image(self, image_path):
        """
        Perform voting with divergence computation between checkpoints and last checkpoint FOR ONLY ONE IMAGE.
        """
        num_checkpoints = len(self.checkpoints)
        
        divergences = np.zeros((num_checkpoints - 1,))
        last_checkpoint = self.checkpoints[-1]
        
        for i in range(num_checkpoints - 1):
            checkpoint = self.checkpoints[i]
            
            iou = calculate_iou_diff_models_batch([image_path], last_checkpoint, checkpoint)
            divergences[i] = iou[0]
        
        evaluation_results = self.weights @ divergences
        return divergences, evaluation_results

    def voting(self, dataloader):
        """
        Perform voting with divergence computation between checkpoints and last checkpoint.
        """
        num_checkpoints = len(self.checkpoints)
        num_images = len(dataloader.dataset)
        
        divergences = np.zeros((num_checkpoints - 1, num_images))
        last_checkpoint = self.checkpoints[-1]
        
        for i in range(num_checkpoints - 1):
            checkpoint = self.checkpoints[i]
            
            for j, batch in enumerate(dataloader):
                iou = calculate_iou_diff_models_batch(batch, last_checkpoint, checkpoint)
                print(iou)
                divergences[i, j * len(batch):j * len(batch) + len(batch)] = iou
        
        evaluation_results = self.weights @ divergences
        return divergences, evaluation_results
    
    def _process_checkpoint_shared(self, i, checkpoint, last_checkpoint, dataloader, shared_divergences, num_images):
        """
        Helper function for parallel processing of each checkpoint's divergence calculation.
        """
        divergence = []
        for batch in dataloader:
            iou = calculate_iou_diff_models_batch(batch, last_checkpoint, checkpoint)
            divergence.append(iou)
        shared_divergences[i * num_images:(i + 1) * num_images] = np.concatenate(divergence)

    def voting_parallelized(self, dataloader):
        """
        Parallelized voting with shared memory divergence storage.
        """
        num_checkpoints = len(self.checkpoints)
        num_images = len(dataloader.dataset)
        
        # Shared memory for divergences array
        shared_divergences = Array(c_double, (num_checkpoints - 1) * num_images)
        
        # Shared data for multiprocessing
        manager = Manager()
        shared_checkpoints = manager.list(self.checkpoints)
        shared_dataloader = manager.list(dataloader)

        args = [
            (i, shared_checkpoints[i], shared_checkpoints[-1], shared_dataloader, shared_divergences, num_images)
            for i in range(num_checkpoints - 1)
        ]

        # Using multiprocessing pool for parallel processing
        with Pool() as pool:
            pool.starmap(self._process_checkpoint_shared, args)

        # Reshape shared_divergences into final form after processing
        divergences = np.frombuffer(shared_divergences.get_obj()).reshape((num_checkpoints - 1, num_images))

        # Calculate weighted sum of divergences
        evaluation_results = self.weights @ divergences

        return divergences, evaluation_results
    
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
    coco_dataloader = DataLoader(dataset=pytorch_dataset, batch_size=batch_size, shuffle=False)
    
    return coco_dataloader

if __name__ == '__main__':
    persentage_of_lowest = 0.2
    n_images = 512
    n_checkpoints = 20

    ensemble = EnsembleOfCheckpoints(folder_path='./data/checkpoints/AdamW')
    dataloader = get_coco_dataloader(n_images)
    
    divergences, eval = ensemble.voting(dataloader)
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
