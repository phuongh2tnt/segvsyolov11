import sys
import torch
import argparse
import numpy as np
import os
import torchvision.transforms as T
from scipy.ndimage import label, find_objects
from PIL import Image, ImageDraw, ImageFont
from sklearn import metrics
from timeit import default_timer as timer
from utils.iris_dataset import visualize
import utils.metric2 as custom_metrics  # Import your custom metrics

sys.path.append(os.path.abspath('/content/segatten/train'))
from unetse import Unet

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

def predict(in_file, img_size=480):
    """
    :param in_file: image file
    :param img_size: image size
    """
    model.eval()

    # Pre-process input image
    img = Image.open(in_file).convert('RGB')  # convert to RGB image
    W, H = img.size
    img_resized = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)(img)
    img_tensor = T.ToTensor()(img_resized).to(device, dtype=torch.float).unsqueeze(dim=0)

    # Perform a forward pass
    logits = model(img_tensor)
    logits = torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)(logits)

    # Measure the segmentation performance
    seg_map = logits.cpu().detach().numpy().argmax(axis=1)
    seg_map = seg_map.squeeze()  # 'squeeze' used to remove the first dimension of 1 (i.e., batch size)

    # Count the number of segments (connected components)
    labeled_seg_map, num_segments = label(seg_map)
    print(f"Number of segments: {num_segments}")

    # Calculate sizes of each segment
    segment_sizes = np.bincount(labeled_seg_map.flatten())
    segment_sizes[0] = 0  # The background (0) is not counted

    # Calculate thresholds
    avg_size = np.mean(segment_sizes[1:])
    
    # Find segments above average size
    above_avg_sizes = [size for size in segment_sizes[1:] if size > avg_size]
    if above_avg_sizes:
        avg_size_above_avg = np.mean(above_avg_sizes)
    else:
        avg_size_above_avg = 0  # In case there are no segments above average size

    print(f"Avg Size: {avg_size}, Avg Size of Segments Above Average: {avg_size_above_avg}")

    # Create an overlay image
    overlay_image = Image.new('RGB', (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(overlay_image)

    # Assign colors based on size thresholds and draw on overlay
    for label_val in np.unique(labeled_seg_map):
        if label_val == 0:
            continue  # Skip background
        size = segment_sizes[label_val]
        if size <= avg_size:
            color = (0, 255, 0)  # Green for average or below
        else:
            color = (0, 0, 255)  # Blue for above average
        
        # Draw the segments with the chosen color
        segment_mask = (labeled_seg_map == label_val)
        overlay_image_np = np.array(overlay_image)
        overlay_image_np[segment_mask] = color
        overlay_image = Image.fromarray(overlay_image_np)

    # Blend the overlay image with the original image
    blended_image = Image.blend(img, overlay_image, alpha=0.5)

    # Add text to the blended image
    draw = ImageDraw.Draw(blended_image)
    standard_font = ImageFont.truetype("/content/segatten/test/Arial.ttf", size=100)  # Adjust path if necessary

    model_text = f"Model: {cmd_args.net}"
    segment_text = f"Số lượng tôm: {num_segments}"
    avg_size_text = f"Avg Size Above Avg: {avg_size_above_avg:.2f}"

    # Position the texts
    draw.text((10, 10), model_text, fill=(255, 255, 255), font=standard_font)
    draw.text((10, 120), segment_text, fill=(255, 255, 255), font=standard_font)
    draw.text((10, 230), avg_size_text, fill=(255, 255, 255), font=standard_font)

    # Save the final blended image
    blended_image.save(cmd_args.output + os.sep + os.path.basename(in_file))
    print(f'File: {os.path.basename(in_file)} done. Số lượng tôm: {num_segments}')

    return seg_map  # Return the segmentation map for metric calculation



if __name__ == "__main__":

    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Test a post-mortem iris segmentation model')
    args.add_argument('-i', '--input', default=None, type=str, help='Path to the input text file containing image names')
    args.add_argument('-w', '--weights', default='/content/drive/My Drive/AI/udeep/unetse/unetse_epoch_71_iou_0.8435.pt', type=str,
                      help='Trained weights')
    args.add_argument('-o', '--output', default='outputs', type=str, help='Output folder')
    args.add_argument('-m', '--metrics_output', default='metrics.txt', type=str, help='File to save the metrics')
    args.add_argument('-n','--net',default='unetse',type=str,help='create model')
    cmd_args = args.parse_args()

    device = setup_cuda()

    # 2. Create a segmentation model, then load the trained weights
    if cmd_args.net == 'unetse':
        model = Unet(in_ch=3, out_ch=2).to(device)
    elif cmd_args.net == 'unetres':
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet18', in_channels=3, classes=2).to(device)
    model.load_state_dict(torch.load(cmd_args.weights, device))
    print('The segmentation model has been loaded.')

    # 3. Read the image names from the input text file
    with open(cmd_args.input, 'r') as f:
        image_files = [line.strip() + '.jpg' for line in f.readlines()]

    # Initialize metrics accumulators
    all_metrics = {
        'F1': [],
        'IOU': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': []
    }

    # 4. Open the metrics output file
    with open(cmd_args.metrics_output, 'w') as metrics_file:
        # 5. Perform segmentation and calculate metrics for each image
        for image_file in image_files:
            image_path = os.path.join('/content/segatten/test/inputs', image_file)
            seg_map = predict(image_path)

            # Assume that you have a ground truth mask for each image in the same folder
            # with "_mask" appended to the image name before the extension
            ground_truth_path = image_path.replace(".jpg", ".png")

            # Load the ground truth mask
            gt = Image.open(ground_truth_path).convert('L')
            gt = np.array(gt)

            # Calculate metrics
            f1 = custom_metrics.f1(seg_map, gt)
            iou = custom_metrics.iou(seg_map, gt)
            accuracy = custom_metrics.accuracy(seg_map, gt)
            precision = metrics.precision_score(gt.flatten(), seg_map.flatten(), average='binary', zero_division=1)
            recall = metrics.recall_score(gt.flatten(), seg_map.flatten(), average='binary', zero_division=1)

            # Append metrics to accumulators
            all_metrics['F1'].append(f1)
            all_metrics['IOU'].append(iou)
            all_metrics['Accuracy'].append(accuracy)
            all_metrics['Precision'].append(precision)
            all_metrics['Recall'].append(recall)

            # Write the metrics for this image to the file
            metrics_file.write(f"Image: {image_file}\n")
            metrics_file.write(f"F1 Score: {f1}\n")
            metrics_file.write(f"IOU: {iou}\n")
            metrics_file.write(f"Accuracy: {accuracy}\n")
            metrics_file.write(f"Precision: {precision}\n")
            metrics_file.write(f"Recall: {recall}\n")
            metrics_file.write("\n")

        # 6. Compute and write average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        metrics_file.write("Average Metrics:\n")
        for key, value in avg_metrics.items():
            metrics_file.write(f"{key}: {value}\n")

    print(f"Metrics have been saved to {cmd_args.metrics_output}")
