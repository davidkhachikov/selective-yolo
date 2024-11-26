import os
from ultralytics import YOLO
from utils import calculate_iou_diff_models_batch

from utils_new import calculate_stat_models_batch, calculate_stat_model_batch
from multiprocessing import Manager, Pool, Array
import numpy as np
from torch.utils.data import DataLoader, Dataset
import fiftyone.zoo as foz
from ctypes import c_double
import json

class DatasetOfPaths(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.filepaths[idx]

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
        
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm1d(hidden_dim1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class EnsembleOfCheckpoints:
    def __init__(self, n_checkpoints=20, step=5, start=0, folder_path=".\\checkpoints", prefix="epoch", yolo_model="yolo11n.pt", v2 = True):
        self.n_checkpoints = n_checkpoints
        self.step = step
        self.folder_path = folder_path
        self.prefix = prefix
        self.start = start
        self.checkpoints = self.load_checkpoints()
        self.weights = self.get_weights(len(self.checkpoints) - 1)
        self.thresh = 0.5
        self.yolo_model = yolo_model
        self.yolo = YOLO(yolo_model)
        self.v2 = v2
        
    def save(self, path):
        """
        Saves the ensemble to a specified path, including YOLO model paths, 
        classifier weights, and ensemble metadata.
        """
        data_to_save = {
            "classifier_state_dict": self.classifier.state_dict(),
            "weights": self.weights.tolist(),
            "thresh": self.thresh,
            "n_checkpoints": self.n_checkpoints,
            "step": self.step,
            "start": self.start,
            "folder_path": self.folder_path,
            "prefix": self.prefix,
            "yolo_model": self.yolo_model,
            "v2": self.v2
        }

        # Save the data to a file
        save_file = path if path.endswith(".pth") else f"{path}.pth"
        torch.save(data_to_save, save_file)

    @classmethod
    def load(cls, path):
        """
        Loads the ensemble from a saved file.
        """
        save_file = path if path.endswith(".pth") else f"{path}.pth"
        saved_data = torch.load(save_file)

        ensemble = cls(
            n_checkpoints=saved_data["n_checkpoints"],
            step=saved_data["step"],
            start=saved_data["start"],
            folder_path=saved_data["folder_path"],
            prefix=saved_data["prefix"],
            v2=saved_data["v2"]
        )
        ensemble.classifier = Classifier(12 * (saved_data["n_checkpoints"] - 1), 64, 128) if saved_data["v2"] else Classifier((saved_data["n_checkpoints"] - 1), 64, 128)
        ensemble.classifier.load_state_dict(saved_data["classifier_state_dict"])

        ensemble.yolo_model = saved_data["yolo_model"]
        ensemble.yolo = YOLO(saved_data["yolo_model"])

        ensemble.weights = np.array(saved_data["weights"])
        ensemble.thresh = saved_data["thresh"]

        return ensemble

    def _compute_metrics(self, y_true, y_pred):
        """
        Private method to compute precision, recall, and F1 score.
        """
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return precision, recall, f1

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
                divergences[i, j * len(batch):j * len(batch) + len(batch)] = iou
        
        return divergences.T

    def voting_new(self, dataloader):
        num_checkpoints = len(self.checkpoints)
        num_images = len(dataloader.dataset)
        
        divergences = [[None for _ in range(num_images)] for _ in range(num_checkpoints - 1)]
        last_checkpoint = self.checkpoints[-1]
        
        for i in range(num_checkpoints - 1):
            checkpoint = self.checkpoints[i]
            checkpoint_next = self.checkpoints[i + 1]
            st = 0
            for j, batch in enumerate(dataloader):
                div_stat = calculate_stat_models_batch(batch, last_checkpoint, checkpoint)
                div_stat_next = calculate_stat_models_batch(batch, checkpoint_next, checkpoint)
                stat_checkpoint = calculate_stat_model_batch(batch, checkpoint)
                iou = calculate_iou_diff_models_batch(batch, last_checkpoint, checkpoint)
                for k in range(len(div_stat)):
                    div_stat[k].append(iou[k])
                    div_stat[k].extend(div_stat_next[k])
                    div_stat[k].extend(stat_checkpoint[k])
                divergences[i][st:st + len(batch)] = div_stat
                st += len(batch)
        divergences = np.array(divergences).transpose(1, 0, 2)
        shape = divergences.shape
        new_shape = shape[:-2] + (shape[-2] * shape[-1],)
        divergences = divergences.reshape(new_shape)
        return divergences

    def train(self, data_pos, data_neg, plot=False, split_size=0.9, epochs=500):
        """
        Public method to train the model, save the best model based on validation loss,
        and optionally plot training and validation metrics.
        """
        if not (0.0 < split_size < 1.0):
            raise ValueError("split_size must be a float between 0 and 1 (exclusive).")
        
        print("Starting calculation of divergences between non-noisy images")
        pos_divergences = self.voting_new(data_pos) if self.v2 else self.voting(data_pos)
        print("Starting calculating of divergences between noisy images")
        neg_divergences = self.voting_new(data_pos) if self.v2 else self.voting(data_neg)
    
        n_pos = pos_divergences.shape[0]
        n_neg = neg_divergences.shape[0]
        print(pos_divergences.shape)
        X = np.vstack([
            pos_divergences, 
            neg_divergences
        ])
        y = [1] * n_pos + [0] * n_neg
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=(1 - split_size), random_state=42, stratify=y
        )
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        print("Starting model training: ")
        self.classifier = Classifier(X_train.shape[1], 64, 128)
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=0.01)
        loss_fn = torch.nn.BCELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []
        precision_history = []
        recall_history = []
        f1_history = []
        print(X_train_tensor.shape)
        
        for epoch in range(epochs):
            # Training step
            self.classifier.train()
            optimizer.zero_grad()
            
            outputs = self.classifier(X_train_tensor)
            train_loss = loss_fn(outputs, y_train_tensor.view(-1, 1))
            train_loss.backward()
            optimizer.step()
            
            # Calculate training metrics
            with torch.no_grad():
                train_preds = (outputs > self.thresh).float()
                train_labels = y_train_tensor.view(-1, 1).float()
                train_acc = (train_preds == train_labels).sum().item() / len(y_train_tensor)
                
                train_preds_np = train_preds.cpu().numpy()
                train_labels_np = train_labels.cpu().numpy()
                
                train_precision, train_recall, train_f1 = self._compute_metrics(train_labels_np, train_preds_np)
            
            # Validation step
            self.classifier.eval()
            with torch.no_grad():
                val_outputs = 1 - self.classifier(X_val_tensor)
                val_loss = loss_fn(val_outputs, y_val_tensor.view(-1, 1))
                
                val_preds = (val_outputs > self.thresh).float()
                val_labels = y_val_tensor.view(-1, 1).float()
                val_acc = (val_preds == val_labels).sum().item() / len(y_val_tensor)
                
            # Save the best model based on validation loss
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = self.classifier.state_dict()  # Save the best model state
            
            # Track statistics
            train_loss_history.append(train_loss.item())
            val_loss_history.append(val_loss.item())
            train_accuracy_history.append(train_acc)
            val_accuracy_history.append(val_acc)
            precision_history.append(train_precision)
            recall_history.append(train_recall)
            f1_history.append(train_f1)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}, "
                      f"Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                      f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
                      f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        
        # Update self.classifier with the best model
        if best_model_state:
            self.classifier.load_state_dict(best_model_state)
            print(f"Best model loaded with validation loss: {best_val_loss:.4f}")
        
        # Plot training and validation statistics
        if plot:
            plt.figure(figsize=(14, 10))
            
            # Plot loss
            plt.subplot(2, 2, 1)
            plt.plot(range(epochs), train_loss_history, label='Train Loss', color='blue')
            plt.plot(range(epochs), val_loss_history, label='Validation Loss', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs')
            plt.legend()
            
            # Plot accuracy
            plt.subplot(2, 2, 2)
            plt.plot(range(epochs), train_accuracy_history, label='Train Accuracy', color='green')
            plt.plot(range(epochs), val_accuracy_history, label='Validation Accuracy', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy over Epochs')
            plt.legend()
            
            # Plot precision
            plt.subplot(2, 2, 3)
            plt.plot(range(epochs), precision_history, label='Precision', color='purple')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.title('Precision over Epochs')
            plt.legend()
            
            # Plot recall
            plt.subplot(2, 2, 4)
            plt.plot(range(epochs), recall_history, label='Recall', color='brown')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('Recall over Epochs')
            plt.legend()
            
            # Adjust layout and save the figure
            plt.tight_layout()
            plt.savefig('training_validation_metrics.png')
            plt.show()
        
        return
    
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
        
        for i in range(self.start, self.start + self.n_checkpoints * self.step, self.step):
            checkpoint_name = f"{self.prefix}{i}.pt"
            checkpoint_path = os.path.join(self.folder_path, checkpoint_name)
            
            print(f"Loading checkpoint: {checkpoint_path}")
            loaded_checkpoints.append(YOLO(checkpoint_path).to(device))
        
        return loaded_checkpoints
    
    def vote_one_image(self, image_path):
        """
        Perform voting with divergence computation between checkpoints and last checkpoint FOR ONLY ONE IMAGE.
        """
        self.classifier.eval()
        dataloader = DataLoader(dataset=DatasetOfPaths([image_path]), batch_size=1, shuffle=False)
        
        divergences = self.voting_new(dataloader) if self.voting_new else self.voting(dataloader)
        
        div_tensor = torch.FloatTensor(divergences)
        pred = self.classifier(div_tensor)[0]
        pred = 1 - pred if self.v2 else pred
        return pred > self.thresh, pred, self.yolo(image_path)
    
def get_coco_dataloader(n_images=64, batch_size=64):
    coco_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="test",
        max_samples=int(n_images),
        shuffle=False,
    )

    pytorch_dataset = DatasetOfPaths([sample.filepath for sample in coco_dataset])

    coco_dataloader = DataLoader(dataset=pytorch_dataset, batch_size=batch_size, shuffle=False)
    
    return coco_dataloader
