from dataclasses import dataclass

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from ultralytics import YOLO

from incident_investigator.ingestion import Window

CLASSES = ["fighting", "vandalism", "fire_smoke", "normal"]

# phrased around what the camera would actually show, not the abstract label —
# CLIP matches scene description a lot better than bare category words
PROMPTS = {
    "fighting": "surveillance footage of two or more people physically fighting, punching, or grappling",
    "vandalism": "surveillance footage of a person spray painting graffiti or breaking and damaging property",
    "fire_smoke": "surveillance footage of a fire with visible flames or thick smoke",
    "normal": "surveillance footage of an ordinary street or hallway with people calmly walking or standing",
}

PERSON_CLASS_ID = 0  # COCO index for "person"

try:
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _clip_model.eval()
except Exception as e:
    raise RuntimeError(f"failed to load CLIP ViT-B-32 (openai weights): {e}") from e

with torch.no_grad():
    _text_tokens = _clip_tokenizer([PROMPTS[c] for c in CLASSES])
    _text_features = _clip_model.encode_text(_text_tokens)
    _text_features = _text_features / _text_features.norm(dim=-1, keepdim=True)

try:
    _yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    raise RuntimeError(f"failed to load YOLOv8n weights: {e}") from e


@dataclass
class WindowFeatures:
    class_scores: dict
    person_count: int
    motion_magnitude: float


def _clip_scores(frames):
    images = torch.stack([_clip_preprocess(Image.fromarray(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB))) for f in frames])
    with torch.no_grad():
        image_features = _clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        window_feature = image_features.mean(dim=0, keepdim=True)
        window_feature = window_feature / window_feature.norm(dim=-1, keepdim=True)
        logits = 100.0 * window_feature @ _text_features.T
        probs = logits.softmax(dim=-1).squeeze(0)
    return {c: probs[i].item() for i, c in enumerate(CLASSES)}


def _person_count(frames):
    max_count = 0
    for f in frames:
        result = _yolo_model(f.image, classes=[PERSON_CLASS_ID], verbose=False)[0]
        max_count = max(max_count, len(result.boxes))
    return max_count


def _motion_magnitude(frames):
    if len(frames) < 2:
        return 0.0
    grays = [cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) for f in frames]
    magnitudes = []
    for prev, nxt in zip(grays, grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(mag.mean())
    return float(np.mean(magnitudes))


def extract_window_features(window: Window) -> WindowFeatures:
    return WindowFeatures(
        class_scores=_clip_scores(window.frames),
        person_count=_person_count(window.frames),
        motion_magnitude=_motion_magnitude(window.frames),
    )
