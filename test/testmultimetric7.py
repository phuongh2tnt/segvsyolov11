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
