#code này đã làm xong phần ghi chữ vào hình
#Lam xong tung hinh roi
import sys
import torch
import argparse
import numpy as np
import os
import torchvision.transforms as T
from scipy.ndimage import label  # For counting connected components
from utils.iris_dataset import visualize
from PIL import Image, ImageDraw, ImageFont
from timeit import default_timer as timer

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

    # Count the number of segments (connected components)
    labeled_seg_map, num_segments = label(seg_map)
    print(f"Number of segments: {num_segments}")

    # Overlay the segment count on the image
    overlaid = visualize(seg_map, np.array(img))
    overlaid = Image.fromarray(overlaid)

    draw = ImageDraw.Draw(overlaid)
    
    # Load the standard font
    standard_font = ImageFont.truetype("/content/segatten/test/Arial.ttf", size=100)  # Đoạn font chữ này thay đường dẫn nếu thay máy
    # Load the large font for the segment count
    large_font = ImageFont.truetype("/content/segatten/test/Arial.ttf", size=100)  # Larger font size for the number of segments

   # Add text "Model: USEnet"
    model_text = "Model: USEnet"
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

if __name__ == "__main__":

    # 1. Parse the command arguments
    args = argparse.ArgumentParser(description='Test a post-mortem iris segmentation model')
    args.add_argument('-i', '--input', default=None, type=str, help='Path to the input image')
    args.add_argument('-w', '--weights', default='/content/drive/My Drive/AI/udeep/unetse/unetse_epoch_71_iou_0.8435.pt', type=str,
                      help='Trained weights')
    args.add_argument('-o', '--output', default='outputs', type=str, help='Output folder')
    cmd_args = args.parse_args()

    device = setup_cuda()

    # 2. Create a segmentation model, then load the trained weights
    model = Unet(in_ch=3, out_ch=2).to(device)
    model.load_state_dict(torch.load(cmd_args.weights, device))
    print('The segmentation model has been loaded.')

    # 3. Perform segmentation and count segments
    predict(cmd_args.input)
