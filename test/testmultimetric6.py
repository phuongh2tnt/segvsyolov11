import sys
import torch
import argparse
import numpy as np
import os
import torchvision.transforms as T
from scipy.ndimage import label
from PIL import Image, ImageDraw, ImageFont
import cv2  # Import OpenCV for contour detection
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

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

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

    # Convert segmentation map to binary image
    seg_map_binary = np.uint8(seg_map * 255)  # Assuming shrimp are labeled as 1

    # Perform distance transform
    dist_transform = distance_transform_edt(seg_map_binary)

    # Threshold the distance transform to get sure foreground areas
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Use markers for the watershed algorithm
    ret, markers = cv2.connectedComponents(sure_fg)

    # Apply watershed to segment the connected components
    markers = markers + 1
    markers[seg_map_binary == 0] = 0
    markers = cv2.watershed(np.array(img), markers)

    # Count the shrimp based on the watershed result
    num_shrimp = len(np.unique(markers)) - 1  # subtract one to exclude the background

    print(f"Number of shrimp: {num_shrimp}")

    # Create an overlay image
    overlay_image = Image.new('RGB', (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(overlay_image)

    # Draw contours on the overlay image
    overlay_image_np = np.array(overlay_image)
    for marker in np.unique(markers):
        if marker == -1:
            continue  # Skip the boundary marker
        mask = np.uint8(markers == marker)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image_np, contours, -1, (0, 255, 0), thickness=cv2.FILLED)

    overlay_image = Image.fromarray(overlay_image_np)

    # Blend the overlay image with the original image
    blended_image = Image.blend(img, overlay_image, alpha=0.5)

    # Add text to the blended image
    draw = ImageDraw.Draw(blended_image)
    standard_font = ImageFont.truetype("/content/segatten/test/Arial.ttf", size=100)  # Adjust path if necessary
    large_font = ImageFont.truetype("/content/segatten/test/Arial.ttf", size=100)  # Larger font size for the number of segments

    model_text = f"Model: {cmd_args.net}"
    shrimp_text = f"Số lượng tôm: {num_shrimp}"

    # Get the bounding box of the model text
    model_text_bbox = draw.textbbox((0, 0), model_text, font=standard_font)
    # Get the bounding box of the shrimp count text
    shrimp_text_bbox = draw.textbbox((0, 0), shrimp_text, font=large_font)

    # Calculate text width and height
    model_text_width = model_text_bbox[2] - model_text_bbox[0]
    model_text_height = model_text_bbox[3] - model_text_bbox[1]
    shrimp_text_width = shrimp_text_bbox[2] - shrimp_text_bbox[0]
    shrimp_text_height = shrimp_text_bbox[3] - shrimp_text_bbox[1]

    # Position the texts
    model_text_position = (W - model_text_width - 10, H - model_text_height - shrimp_text_height - 20)
    shrimp_text_position = (W - shrimp_text_width - 10, H - shrimp_text_height - 10)

    draw.text(model_text_position, model_text, fill=(255, 255, 255), font=standard_font)
    draw.text(shrimp_text_position, shrimp_text, fill=(255, 255, 255), font=large_font)

    # Save the final blended image
    blended_image.save(cmd_args.output + os.sep + os.path.basename(in_file))
    print(f'File: {os.path.basename(in_file)} done. Số lượng tôm: {num_shrimp}')

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
