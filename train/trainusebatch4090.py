import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import argparse
from torch.amp import autocast, GradScaler
from utils.metric2 import accuracy, iou, f1, precision, recall
import utils.metrics as metrics

# Import your UNet class
from unetse import Unet  # Assuming your UNet code is in a file named unet_with_se.py

# Setup CUDA
def setup_cuda():
    # Setting seeds for reproducibility
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(accumulation_steps=2):
    """
    Train the model over a single epoch
    :return: training loss and segmentation performance
    """
    model.train()
    train_loss = 0.0
    performance = 0
    train_metrics = {'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    scaler = GradScaler('cuda')
    for i, (img, gt) in enumerate(tqdm(train_loader, ncols=80, desc='Training')):
        optimizer.zero_grad()
        img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        with autocast('cuda'):
            logits = model(img)
            loss = loss_fn(logits, gt) / accumulation_steps
        
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
       
        train_loss += loss.item()
        seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
        prediction = logits.argmax(axis=1).cpu().numpy()
        gt = gt.cpu().detach().numpy()
        performance += getattr(metrics, cmd_args.metric)(seg_maps, gt)
        train_metrics['iou'] += iou(prediction, gt)
        train_metrics['accuracy'] += accuracy(prediction, gt)
        train_metrics['precision'] += precision(prediction, gt)
        train_metrics['recall'] += recall(prediction, gt)
        train_metrics['f1'] += f1(prediction, gt)

    for key in train_metrics:
        train_metrics[key] /= len(train_loader)
    return train_loss / len(train_loader), performance / len(train_loader), train_metrics

def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and segmentation performance
    """
    model.eval()
    valid_loss = 0.0
    performance = 0
    val_metrics = {'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    with torch.no_grad():
        for i, (img, gt) in enumerate(valid_loader):
            img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
            with autocast('cuda'):
                logits = model(img)
                loss = loss_fn(logits, gt)
            valid_loss += loss.item()
            seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
            prediction = logits.argmax(axis=1).cpu().numpy()
            gt = gt.cpu().detach().numpy()
            performance += getattr(metrics, cmd_args.metric)(seg_maps, gt)
            val_metrics['iou'] += iou(prediction, gt)
            val_metrics['accuracy'] += accuracy(prediction, gt)
            val_metrics['precision'] += precision(prediction, gt)
            val_metrics['recall'] += recall(prediction, gt)
            val_metrics['f1'] += f1(prediction, gt)

    for key in val_metrics:
        val_metrics[key] /= len(valid_loader)

    return valid_loss / len(valid_loader), performance / len(valid_loader), val_metrics

def save_metrics_to_file(train_history, val_history, filename):
    with open(filename, 'w') as f:
        # Write headers
        headers = ['Epoch', 'Train Loss', 'Train IOU', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1',
                   'Val Loss', 'Val IOU', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1']
        f.write('\t'.join(headers) + '\n')
        
        # Write metrics for each epoch
        for epoch in range(len(train_history['loss'])):
            line = [str(epoch)]
            line.append(f"{train_history['loss'][epoch]:.4f}")
            line.append(f"{train_history['iou'][epoch]:.4f}")
            line.append(f"{train_history['accuracy'][epoch]:.4f}")
            line.append(f"{train_history['precision'][epoch]:.4f}")
            line.append(f"{train_history['recall'][epoch]:.4f}")
            line.append(f"{train_history['f1'][epoch]:.4f}")
            line.append(f"{val_history['loss'][epoch]:.4f}")
            line.append(f"{val_history['iou'][epoch]:.4f}")
            line.append(f"{val_history['accuracy'][epoch]:.4f}")
            line.append(f"{val_history['precision'][epoch]:.4f}")
            line.append(f"{val_history['recall'][epoch]:.4f}")
            line.append(f"{val_history['f1'][epoch]:.4f}")
            f.write('\t'.join(line) + '\n')
        
        # Calculate and write averages
        avg_train_loss = np.mean(train_history['loss'])
        avg_train_iou = np.mean(train_history['iou'])
        avg_train_accuracy = np.mean(train_history['accuracy'])
        avg_train_precision = np.mean(train_history['precision'])
        avg_train_recall = np.mean(train_history['recall'])
        avg_train_f1 = np.mean(train_history['f1'])
        
        avg_val_loss = np.mean(val_history['loss'])
        avg_val_iou = np.mean(val_history['iou'])
        avg_val_accuracy = np.mean(val_history['accuracy'])
        avg_val_precision = np.mean(val_history['precision'])
        avg_val_recall = np.mean(val_history['recall'])
        avg_val_f1 = np.mean(val_history['f1'])
        
        f.write('\nAverages:\n')
        avg_line = ['Averages']
        avg_line.append(f"{avg_train_loss:.4f}")
        avg_line.append(f"{avg_train_iou:.4f}")
        avg_line.append(f"{avg_train_accuracy:.4f}")
        avg_line.append(f"{avg_train_precision:.4f}")
        avg_line.append(f"{avg_train_recall:.4f}")
        avg_line.append(f"{avg_train_f1:.4f}")
        avg_line.append(f"{avg_val_loss:.4f}")
        avg_line.append(f"{avg_val_iou:.4f}")
        avg_line.append(f"{avg_val_accuracy:.4f}")
        avg_line.append(f"{avg_val_precision:.4f}")
        avg_line.append(f"{avg_val_recall:.4f}")
        avg_line.append(f"{avg_val_f1:.4f}")
        f.write('\t'.join(avg_line) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a deep model for shrimp segmentation')
    parser.add_argument('-d', '--dataset', default="E:/thanh/ntu_group/phuong/segatten/train/dataset", type=str, help='Dataset folder')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('-i', '--img-size', default=480, type=int, help='Image size')
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, help='Checkpoint folder')
    parser.add_argument('-t', '--metric', default='iou', type=str, help='Metric for optimization')

    cmd_args = parser.parse_args()
    device = setup_cuda()

    # 2. Load the dataset
    from utils.lanedatasetv4 import LaneDataset

    train_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='test', img_size=cmd_args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cmd_args.batch_size,
                                               shuffle=True,
                                               num_workers=6)

    valid_dataset = LaneDataset(dataset_dir=cmd_args.dataset, subset='valid', img_size=cmd_args.img_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cmd_args.batch_size,
                                               shuffle=False,
                                               num_workers=6)

    # 3. Create a segmentation model using UNet with SE layers
    model = Unet(in_ch=3, out_ch=2).to(device)

    # 4. Specify loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_history = {'loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    val_history = {'loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    # 5. Start training the model
    max_perf = 0
    for epoch in range(cmd_args.epochs):
        # 5.1. Train the model over a single epoch
        train_loss, train_perf, train_metrics = train_model()

        # 5.2. Validate the model
        val_loss, valid_perf, val_metrics = validate_model()

        print('Epoch: {} \tTraining {}: {:.4f} \tValid {}: {:.4f}'.format(epoch, cmd_args.metric, train_perf,
                                                                          cmd_args.metric, valid_perf))

        train_history['loss'].append(train_loss)
        val_history['loss'].append(val_loss)
        train_history['iou'].append(train_metrics['iou'])
        val_history['iou'].append(val_metrics['iou'])
        train_history['f1'].append(train_metrics['f1'])
        val_history['f1'].append(val_metrics['f1'])
        train_history['precision'].append(train_metrics['precision'])
        val_history['precision'].append(val_metrics['precision'])
        train_history['recall'].append(train_metrics['recall'])
        val_history['recall'].append(val_metrics['recall'])
        train_history['accuracy'].append(train_metrics['accuracy'])
        val_history['accuracy'].append(val_metrics['accuracy'])
        # 5.3. Save the model if the validation performance is increasing
        path = "E:/thanh/ntu_group/phuong/segatten/train/checkpoints"
        path2 = "E:/thanh/ntu_group/phuong/segatten/train/graph"
        if valid_perf > max_perf:
            print('Valid {} increased ({:.4f} --> {:.4f}). Model saved'.format(cmd_args.metric, max_perf, valid_perf))
            torch.save(model.state_dict(), f"{path}/unetse_epoch_{epoch}_{cmd_args.metric}_{valid_perf:.4f}.pt")
            max_perf = valid_perf
    # 6. Plot and save training and validation metrics
    epochs_range = range(cmd_args.epochs)
    for metric_name in train_history:
        plt.figure()
        plt.plot(epochs_range, train_history[metric_name], label=f'Training {metric_name}')
        plt.plot(epochs_range, val_history[metric_name], label=f'Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} vs. Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{path2}/unetse_{metric_name}.png")
        plt.show()

    # Save metrics to file
    save_metrics_to_file(train_history, val_history, "metrics_use8.txt")
