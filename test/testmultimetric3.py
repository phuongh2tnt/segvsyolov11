import sys
import torch
import argparse
import numpy as np
import os
import torchvision.transforms as T
from scipy.ndimage import label, find_objects
from PIL import Image, ImageDraw, ImageFont
import utils.metric2 as metrics
from utils.iris_dataset import visualize

sys.path.append(os.path.abspath('/content/segatten/train'))
from unetse import Unet

def setup_cuda(seed=50):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model(cmd_args, device):
    if cmd_args.net == 'unetse':
        model = Unet(in_ch=3, out_ch=2).to(device)
    elif cmd_args.net == 'unetres':
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet18', in_channels=3, classes=2).to(device)
    else:
        raise ValueError(f"Unsupported model type: {cmd_args.net}")
    
    model.load_state_dict(torch.load(cmd_args.weights, device))
    model.eval()
    print(f'The segmentation model ({cmd_args.net}) has been loaded.')
    return model

def is_rectangle_shape(slice_):
    """
    Determine if the given bounding box slice corresponds to a rectangular shape.
    For simplicity, we can assume that if the bounding box is not too narrow or too elongated, it's a rectangle.
    """
    y_min, y_max = slice_[0].start, slice_[0].stop
    x_min, x_max = slice_[1].start, slice_[1].stop
    height = y_max - y_min
    width = x_max - x_min
    
    # Define aspect ratio limits for what you consider a "rectangle"
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2.0
    
    aspect_ratio = width / height
    return min_aspect_ratio <= aspect_ratio <= max_aspect_ratio

def count_rectangular_segments(seg_map):
    labeled_seg_map, num_segments = label(seg_map)
    slices = find_objects(labeled_seg_map)

    rectangular_segments = 0
    for slice_ in slices:
        if slice_ and is_rectangle_shape(slice_):
            rectangular_segments += 1

    return rectangular_segments

def predict(model, in_file, device, img_size=480):
    img = Image.open(in_file).convert('RGB')
    W, H = img.size
    img_resized = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)(img)
    img_tensor = T.ToTensor()(img_resized).to(device, dtype=torch.float).unsqueeze(dim=0)
    
    with torch.no_grad():
        logits = model(img_tensor)
    
    logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    seg_map = logits.cpu().detach().numpy().argmax(axis=1).squeeze()

    num_rectangular_segments = count_rectangular_segments(seg_map)
    print(f"Number of rectangular segments: {num_rectangular_segments}")

    overlaid = visualize(seg_map, np.array(img))
    overlaid_img = Image.fromarray(overlaid)
    draw = ImageDraw.Draw(overlaid_img)

    add_text_overlay(draw, num_rectangular_segments, cmd_args.net, W, H)

    save_path = os.path.join(cmd_args.output, os.path.basename(in_file))
    overlaid_img.save(save_path)
    print(f'File: {os.path.basename(in_file)} done. Số lượng tôm (rectangular segments): {num_rectangular_segments}')

    return seg_map

def add_text_overlay(draw, num_segments, model_name, W, H):
    font_path = "/content/segatten/test/Arial.ttf"
    
    try:
        standard_font = ImageFont.truetype(font_path, size=50)
        large_font = ImageFont.truetype(font_path, size=100)
    except IOError:
        print("Warning: Font not found, using default font.")
        standard_font = ImageFont.load_default()
        large_font = ImageFont.load_default()
    
    model_text = f"Model: {model_name}"
    segment_text = f"Số lượng tôm: {num_segments}"
    
    model_text_bbox = draw.textbbox((0, 0), model_text, font=standard_font)
    segment_text_bbox = draw.textbbox((0, 0), segment_text, font=large_font)
    
    model_text_position = (W - model_text_bbox[2] - 10, H - model_text_bbox[3] - segment_text_bbox[3] - 20)
    segment_text_position = (W - segment_text_bbox[2] - 10, H - segment_text_bbox[3] - 10)
    
    draw.text(model_text_position, model_text, fill=(255, 255, 255), font=standard_font)
    draw.text(segment_text_position, segment_text, fill=(255, 255, 255), font=large_font)

def calculate_and_log_metrics(image_files, model, device, cmd_args):
    all_metrics = {'F1': [], 'IOU': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

    with open(cmd_args.metrics_output, 'w') as metrics_file:
        for image_file in image_files:
            image_path = os.path.join('/content/segatten/test/inputs', image_file)
            seg_map = predict(model, image_path, device)

            ground_truth_path = image_path.replace(".jpg", ".png")
            gt = np.array(Image.open(ground_truth_path).convert('L'))

            f1 = metrics.f1(seg_map, gt)
            iou = metrics.iou(seg_map, gt)
            accuracy = metrics.accuracy(seg_map, gt)
            precision = metrics.precision(seg_map, gt)
            recall = metrics.recall(seg_map, gt)

            for metric, value in zip(['F1', 'IOU', 'Accuracy', 'Precision', 'Recall'], [f1, iou, accuracy, precision, recall]):
                all_metrics[metric].append(value)
                metrics_file.write(f"{metric}: {value}\n")
            metrics_file.write("\n")
        
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        metrics_file.write("Average Metrics:\n")
        for key, value in avg_metrics.items():
            metrics_file.write(f"{key}: {value}\n")

    print(f"Metrics have been saved to {cmd_args.metrics_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a post-mortem iris segmentation model')
    parser.add_argument('-i', '--input', required=True, type=str, help='Path to the input text file containing image names')
    parser.add_argument('-w', '--weights', required=True, type=str, help='Path to trained weights')
    parser.add_argument('-o', '--output', default='outputs', type=str, help='Output folder')
    parser.add_argument('-m', '--metrics_output', default='metrics.txt', type=str, help='File to save the metrics')
    parser.add_argument('-n', '--net', default='unetse', type=str, help='Specify the model to use')
    cmd_args = parser.parse_args()

    device = setup_cuda()
    model = load_model(cmd_args, device)

    with open(cmd_args.input, 'r') as f:
        image_files = [line.strip() + '.jpg' for line in f.readlines()]

    calculate_and_log_metrics(image_files, model, device, cmd_args)
