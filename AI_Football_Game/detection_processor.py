import cv2
import numpy as np
import torch
import supervision as sv

class DetectionProcessor:
    def __init__(self):
        self.confidence_thresholds = {
            0: 0.45,  # ball
            1: 0.51,  # goalkeeper
            2: 0.75,  # player
            3: 0.55,  # referee
        }

    def filter_detections(self, results, ball_results=None):
        boxes = results[0].boxes
        filtered_indices = []
        
        
        if ball_results is not None and len(ball_results[0].boxes) > 0:
        
            main_data = boxes.data.clone()
            
            ball_data = ball_results[0].boxes.data.clone()
            ball_data[:, 5] = 0  
            
            non_ball_mask = main_data[:, 5] != 0
            main_data = main_data[non_ball_mask]
            
            if 0 in self.confidence_thresholds:
                threshold = self.confidence_thresholds[0]
                ball_mask = ball_data[:, 4] >= threshold
                ball_data = ball_data[ball_mask]
            
            
            combined_data = torch.cat([main_data, ball_data], dim=0)
            boxes.data = combined_data
        
        for i in range(len(boxes)):
            class_id = int(boxes.data[i, 5])
            confidence = float(boxes.data[i, 4])
            
            if class_id in self.confidence_thresholds:
                threshold = self.confidence_thresholds[class_id]
                if confidence >= threshold:
                    filtered_indices.append(i)
            else:
                filtered_indices.append(i)
                print(f"Порог уверенности для класса {class_id} не задан, использован по умолчанию")
        
        results[0].boxes.data = boxes.data[filtered_indices]
    
        return results
    
    def smart_detection_tracking(self, detections, tracker):

        class0_mask = detections.class_id == 0
        detections_class0 = detections[class0_mask]
        if len(detections_class0) > 0:
            max_conf_idx = np.argmax(detections_class0.confidence)
            detections_class0 = detections_class0[max_conf_idx:max_conf_idx+1]
            detections_class0.tracker_id = np.array([-1]) 

       
        detections_other = detections[~class0_mask]
        if len(detections_other) > 0:
            detections_other = tracker.update_with_detections(detections_other)
            if not hasattr(detections_other, 'tracker_id'):  
                detections_other.tracker_id = np.arange(len(detections_other))
        
        detections_list = [d for d in [detections_class0, detections_other] if len(d) > 0]

        detections = sv.Detections(
            xyxy=np.concatenate([d.xyxy for d in detections_list]) if detections_list else np.empty((0, 4)),
            confidence=np.concatenate([d.confidence for d in detections_list]) if detections_list else np.empty(0),
            class_id=np.concatenate([d.class_id for d in detections_list]) if detections_list else np.empty(0, dtype=int),
            tracker_id=np.concatenate([d.tracker_id for d in detections_list]) if detections_list else np.empty(0, dtype=int)
        )

        if not hasattr(detections, 'tracker_id'):
            detections.tracker_id = np.empty(0, dtype=int)

        return detections

    def get_keypoints_and_confidences(self, model, frame, conf=0.7):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, save=False, conf=conf, verbose=False)

        if not results or not results[0]:
            return None, None
        
        result = results[0]
        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        return keypoints, confidences

    def filter_points(self, points_match_image, points_field_model):
        try:
            points_match_image = np.array(points_match_image[0])
            points_field_model = np.array(points_field_model)

            mask = np.logical_not(np.all(points_match_image == 0, axis=1))
            points_match_image_filtered = points_match_image[mask].tolist()
            points_field_model_filtered = points_field_model[mask].tolist()

            return points_match_image_filtered, points_field_model_filtered
        
        except Exception as e:
            print(f"Ошибка при фильтрации точек: {e}")
            return [], []