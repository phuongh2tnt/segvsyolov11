import torch
from tqdm import tqdm
import numpy as np
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from utils.metric2 import accuracy, iou, f1, precision, recall 
from torch.amp import GradScaler, autocast
from unet_nat import UNetWithNAT  # Assuming your model is saved in unet_nat.py

# Setup CUDA
def setup_cuda():
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(accumulation_steps=2):
    model.train()
    train_loss = 0.0
    train_metrics = {'iou': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    scaler = GradScaler()

    optimizer.zero_grad()
    for i, (img, gt) in enumerate(tqdm(train_loader, ncols=80, desc='Training')):
        img, gt = img.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        
        with autocast('cuda'):
            logits = model(img)
            loss = loss_fn(logits, gt) / accumulation_steps
        
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accumulation_steps
        seg_maps = logits.cpu().detach().numpy().argmax(axis=1)
        prediction = logits.argmax(axis=1).cpu().numpy()
        gt = gt.cpu().detach().numpy()
        train_metrics['iou'] += iou(prediction, gt)
        train_metrics['accuracy'] += accuracy(prediction, gt)
        train_metrics['precision'] += precision(prediction, gt)
        train_metrics['recall'] += recall(prediction, gt)
        train_metrics['f1'] += f1(prediction, gt)

    for key in train_metrics:
        train_metrics[key] /= len(train_loader)

    return train_loss / len(train_loader), train_metrics

def validate_model():
    model.eval()
    valid_loss = 0.0
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
            val_metrics['iou'] += iou(prediction, gt)
            val_metrics['accuracy'] += accuracy(prediction, gt)
            val_metrics['precision'] += precision(prediction, gt)
            val_metrics['recall'] += recall(prediction, gt)
            val_metrics['f1'] += f1(prediction, gt)

    for key in val_metrics:
        val_metrics[key] /= len(valid_loader)

    return valid_loss / len(valid_loader), val_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a deep model for shrimp segmentation')
    parser.add_argument('-d', '--dataset', default="E:/thanh/ntu_group/phuong/segatten/train/dataset", type=str, help='Dataset folder')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', default=4, type=int, help='Batch size')
    parser.add_argument('-i', '--img-size', default=480, type=int, help='Image size')
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, help='Checkpoint folder')
    parser.add_argument('-t', '--metric', default='iou', type=str, help='Metric for optimization')

    cmd_args = parser.parse_args()
    device = setup_cuda()

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

    model = UNetWithNAT(
        in_channels=3,  # Adapt this to your input channel size
        out_channels=2  # Adapt this to your number of classes
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_history = {'loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    val_history = {'loss': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    metrics_file = "E:/thanh/ntu_group/phuong/segatten/train/metric_unat.txt"

    # Write the header to the TXT file
    with open(metrics_file, mode='w') as file:
        file.write(f"{'Epoch':<10} {'Train Loss':<12} {'Valid Loss':<12} "
                   f"{'Train IoU':<12} {'Valid IoU':<12} "
                   f"{'Train Acc':<12} {'Valid Acc':<12} "
                   f"{'Train Prec':<12} {'Valid Prec':<12} "
                   f"{'Train Recall':<12} {'Valid Recall':<12} "
                   f"{'Train F1':<12} {'Valid F1':<12}\n")

    # Start training the model
    max_perf = 0
    for epoch in range(cmd_args.epochs):
        train_loss, train_metrics = train_model()
        val_loss, val_metrics = validate_model()

        train_perf = train_metrics[cmd_args.metric]
        valid_perf = val_metrics[cmd_args.metric]
        
        print(f'Epoch: {epoch} \tTraining {cmd_args.metric}: {train_perf:.4f} \tValid {cmd_args.metric}: {valid_perf:.4f}')
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

        # Save metrics to TXT file
        with open(metrics_file, mode='a') as file:
            file.write(f"{epoch:<10} {train_loss:<12.4f} {val_loss:<12.4f} "
                       f"{train_metrics['iou']:<12.4f} {val_metrics['iou']:<12.4f} "
                       f"{train_metrics['accuracy']:<12.4f} {val_metrics['accuracy']:<12.4f} "
                       f"{train_metrics['precision']:<12.4f} {val_metrics['precision']:<12.4f} "
                       f"{train_metrics['recall']:<12.4f} {val_metrics['recall']:<12.4f} "
                       f"{train_metrics['f1']:<12.4f} {val_metrics['f1']:<12.4f}\n")

        # Save the model if the validation performance is increasing
        path = "E:/thanh/ntu_group/phuong/segatten/train/checkpoints"
        path2 = "E:/thanh/ntu_group/phuong/segatten/train/graph"
        if valid_perf > max_perf:
            print(f'Valid {cmd_args.metric} increased ({max_perf:.4f} --> {valid_perf:.4f}). Model saved')
            torch.save(model.state_dict(), f"{path}/unetnat_epoch_{epoch}_{cmd_args.metric}_{valid_perf:.4f}.pt")
            max_perf = valid_perf

    # Plot and save training and validation metrics
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
        plt.savefig(f"{path2}/unetnat_{metric_name}.png")
        plt.show()

    # Calculate and print average metrics over all epochs
    avg_metrics = {}
    for key in train_history:
        avg_metrics[f'Train {key}'] = np.mean(train_history[key])
        avg_metrics[f'Valid {key}'] = np.mean(val_history[key])

    print("Average Metrics over all epochs:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")
