"""Segment the subject and composite it over an optional background."""
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

from avatar_paths import MODELS

def subject_layers(neutral, background_path=None):
    height, width = neutral.shape[:2]
    if background_path:
        with Image.open(background_path) as picture:
            fitted = ImageOps.fit(picture.convert('RGB'), (width, height))
        plate = cv2.cvtColor(np.asarray(fitted), cv2.COLOR_RGB2BGR)
    else:
        plate = neutral
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=(MODELS / "selfie_multiclass_256x256.tflite").read_bytes()
        ),
        output_confidence_masks=True,
    )
    rgb = cv2.cvtColor(neutral, cv2.COLOR_BGR2RGB)
    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        mask = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).confidence_masks[0]
        background = cv2.resize(mask.numpy_view(), (width, height))
    support = (background < 0.50).astype(np.uint8)

    joined = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    outlines, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not outlines:
        raise ValueError("На исходном кадре не найден силуэт аватара.")
    silhouette = np.zeros_like(support)
    cv2.drawContours(silhouette, [max(outlines, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    alpha = cv2.GaussianBlur(silhouette, (0, 0), 1.6).astype(np.float32) / 255
    if not background_path:
        # Fill the small strip that can become visible as the subject moves.
        hole = cv2.dilate(silhouette, np.ones((13, 13), np.uint8))
        small_size = (max(1, width // 4), max(1, height // 4))
        fill = cv2.inpaint(cv2.resize(neutral, small_size), cv2.resize(hole, small_size),
                           3, cv2.INPAINT_TELEA)
        plate = np.where(hole[..., None] > 0, cv2.resize(fill, (width, height)), neutral)
    return plate, alpha

