import cv2
from mmdet.apis import inference_detector
import supervision as sv
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames, fps


def detect_frames(frames, model, tracker):
        
    detections = []
    for i in range(0,len(frames)):
        
        result = inference_detector(model, frames[i])
        result = sv.Detections.from_mmdetection(result)
        result = tracker.update_with_detections(result)
        filtered_detections = result[result.confidence >= 0.23]
        detections.append(filtered_detections)
    return detections



def save_video(images, output_path, fps):
    
    if len(images) == 0:
        raise ValueError("Массив изображений не может быть пустым")

    height, width, layers = images[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for image in images:
        out.write(image)
    
    out.release()