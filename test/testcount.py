from PIL import Image, ImageDraw, ImageFont

def predict(in_file, img_size=480):
    # Your existing code ...

    # Count the number of segments (connected components)
    labeled_seg_map, num_segments = label(seg_map)
    print(f"Number of segments: {num_segments}")

    # Overlay the segment count on the image
    overlaid = visualize(seg_map, np.array(img))
    overlaid = Image.fromarray(overlaid)

    draw = ImageDraw.Draw(overlaid)
    font = ImageFont.load_default()
    
    text = f"Segments: {num_segments}"
    
    # Get the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    
    # Calculate text width and height
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Position the text at the bottom-right corner
    text_position = (W - text_width - 10, H - text_height - 10)
    
    draw.text(text_position, text, fill=(255, 255, 255), font=font)

    overlaid.save(cmd_args.output + os.sep + os.path.basename(in_file))
    print('File: ' + os.path.basename(in_file) + ' done')

# The rest of your script remains unchanged
