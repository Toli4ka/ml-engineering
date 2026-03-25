from PIL import Image, ImageDraw, ImageFont


def _format_caption(detection):
    label = detection.get("class_name", str(detection.get("class_id", "object")))
    confidence = detection.get("confidence")
    if confidence is None:
        return label
    return f"{label} {confidence:.2f}"


def draw_detections(image, detections, color=(255, 80, 80), line_width=3):
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()

    for detection in detections:
        x_min, y_min, x_max, y_max = detection["bbox_xyxy"]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)

        caption = _format_caption(detection)
        text_bbox = draw.textbbox((x_min, y_min), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_top = max(0, y_min - text_height - 6)
        text_bottom = text_top + text_height + 4

        draw.rectangle(
            [x_min, text_top, x_min + text_width + 6, text_bottom],
            fill=color,
        )
        draw.text((x_min + 3, text_top + 2), caption, fill="white", font=font)

    return rendered
