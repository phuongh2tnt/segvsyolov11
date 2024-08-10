import sys
import torch
import argparse
import numpy as np
import os
import torchvision.transforms as T
import cv2
from scipy import ndimage as ndi
from scipy.ndimage import label  # For counting connected components
from skimage import morphology
from skimage.segmentation import watershed
from utils.iris_dataset import visualize
from PIL import Image, ImageDraw, ImageFont
import utils.metric2 as metrics  # Import your custom metrics

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

def preprocess_image(image):
    """
    Pre-process the image to enhance object separation.
    :param image: input image as numpy array
    :return: pre-processed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    return processed

def postprocess_segmentation(seg_map):
    """
    Post-process the segmentation map to separate overlapping objects.
    :param seg_map: input segmentation map as numpy array
    :return: post-processed segmentation map
    """
    # Compute the distance transform
    distance_transform = cv2.distanceTransform(seg_map.astype(np.uint8), cv2.DIST_L2, 5)
    
    # Normalize the distance transform
    _, dist_normalized = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, cv2.THRESH_BINARY)
    dist_normalized = np.uint8(dist_normalized)
    
    # Find local maxima
    local_max = morphology.h_maxima(distance_transform, h=1)
    
    # Perform watershed segmentation
    markers = ndi.label(local_max)[0]
    labels = watershed(-distance_transform, markers, mask=seg_map)
    
    return labels

def predict(in_file, img_size=480):
    """
    :param in_file: image file
    :param img_size: image size
    """
    model.eval()

    # Pre-process input image and its ground-truth
    img = Image.open(in_file).convert('RGB')  # convert to RGB image
    W, H = img.size
  
    img_resized = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)(img)
    img_tensor = T.ToTensor()(img_resized).to(device, dtype=torch.float).unsqueeze(dim=0)

    # Perform a forward pass
    logits = model(img_tensor)

    # Upsample the output to the original resolution for a better visualization
    logits = torch.nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)(logits)

    # Measure the segmentation performance
    seg_map = logits.cpu().detach().numpy().argmax(axis=1)
    seg_map = seg_map.squeeze()  # 'squeeze' used to remove the first dimension of 1 (i.e., batch size)

    # Preprocess the image
    preprocessed_img = preprocess_image(np.array(img))

    # Postprocess segmentation map
    seg_map_postprocessed = postprocess_segmentation(seg_map)

    # Count the number of segments (connected components)
    labeled_seg_map, num_segments = label(seg_map_postprocessed)
    print(f"Number of segments: {num_segments}")

    # Overlay the segment count on the image
    overlaid = visualize(seg_map_postprocessed, np.array(img))
    overlaid = Image.fromarray(overlaid)

    draw = ImageDraw.Draw(overlaid)
    
    # Load the standard font
    standard_font = ImageFont.truetype("/content/segatten/test/Arial.ttf", size=100)  # Adjust path if necessary
    # Load the large font for the segment count
    large_font = ImageFont.truetype("/content/segatten/test/Arial.ttf", size=100)  # Larger font size for the number of segments

    # Add text "Model: USEnet"
    model_text = f"Model {cmd_args.net} - Số lượng tôm: {sl}"
    segment_text = f"Số lượng tôm: {num_segments}"
    
    # Get the bounding box of the model text
    model_text_bbox = draw.textbbox((0, 0), model_text, font=standard_font)
    # Get the bounding box of the segment count text
    segment_text_bbox = draw.textbbox((0, 0), segment_text, font=large_font)
    
    # Calculate text width and height
    model_text_width = model_text_bbox[2] - model_text_bbox[0]
    model_text_height = model_text_bbox[3] - model_text_bbox[1]
    segment_text_width = segment_text_bbox[2] - segment_text_bbox[0]
    segment_text_height = segment_text_bbox[3] - segment_text_bbox[1]
    
    # Position the texts
    model_text_position = (W - model_text_width - 10, H - model_text_height - segment_text_height - 20)
    segment_text_position = (W - segment_text_width - 10, H - segment_text_height - 10)
    
    draw.text(model_text_position, model_text, fill=(255, 255, 255), font=standard_font)
    draw.text(segment_text_position, segment_text, fill=(255, 255, 255), font=large_font)

    overlaid.save(cmd_args.output + os.sep + os.path.basename(in_file))
    print(f'File: {os.path.basename(in_file)} done. Số lượng tôm: {num_segments}')

    return seg_map_postprocessed  # Return the post-processed segmentation map for metric calculation

if __name__ == "__main__":

    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Test a post-mortem iris segmentation model')
    args.add_argument('-i', '--input', default=None, type=str, help='Path to the input text file containing image names')
    args.add_argument('-w', '--weights', default='/content/drive/My Drive/AI/udeep/unetse/unetse_epoch_71_iou_0.8435.pt', type=str,
                      help='Trained weights')
    args.add_argument('-o', '--output', default='outputs', type=str, help='Output folder')
    args.add_argument('-m', '--metrics_output', default='metrics.txt', type=str, help='File to save the metrics')
    args.add_argument('-n', '--net', default='unetse', type=str, help='Create model')
    cmd_args = args.parse_args()

    device = setup_cuda()

    # 2. Create a segmentation model, then load the trained weights
    # Change to test all model just 1 code
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
            f1 = metrics.f1(seg_map, gt)
            iou = metrics.iou(seg_map, gt)
            accuracy = metrics.accuracy(seg_map, gt)
            precision = metrics.precision(seg_map, gt)
            recall = metrics.recall(seg_map, gt)

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

        # 6. Compute average metrics
        avg_metrics = {key: np.mean(value) for key, value in all_metrics.items()}
        
        with open(cmd_args.metrics_output, 'a') as metrics_file:
            metrics_file.write("Average Metrics:\n")
            metrics_file.write(f"F1 Score: {avg_metrics['F1']}\n")
            metrics_file.write(f"IOU: {avg_metrics['IOU']}\n")
            metrics_file.write(f"Accuracy: {avg_metrics['Accuracy']}\n")
            metrics_file.write(f"Precision: {avg_metrics['Precision']}\n")
            metrics_file.write(f"Recall: {avg_metrics['Recall']}\n")

    print(f"Metrics have been saved to {cmd_args.metrics_output}")
