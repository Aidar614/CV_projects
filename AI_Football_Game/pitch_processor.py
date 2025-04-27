import cv2
import numpy as np
from field import Field

class PitchProcessor:
    def __init__(self):
        self.field = Field()

    def calculate_homography(self, points_match_image, points_field_model, 
                            previous_homography=None, smoothing_factor=0.5):
        try:
            src_pts = np.float32(points_match_image).reshape(-1, 1, 2)
            dst_pts = np.float32(points_field_model).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography is not None:
                if previous_homography is not None:
                    homography = (1 - smoothing_factor) * previous_homography + smoothing_factor * homography
                return homography
            else:
                print("Не удалось вычислить гомографию.")
                return previous_homography
        except:
            return previous_homography

    def transform_player_coordinates(self, player_points, homography_matrix):
        try:
            track_ids = []
            points = []
            class_ids = []
            for track_id, point, class_id in player_points:
                track_ids.append(track_id)
                points.append(point)
                class_ids.append(class_id)

            points_np = np.float32(points).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(points_np, homography_matrix)
            transformed_points = transformed_points.reshape(-1, 2).tolist()
            
            return list(zip(track_ids, transformed_points, class_ids))
        except Exception as e:
            print(f"Ошибка при трансформации координат: {e}")
            return None

    def get_centers_from_detections(self, detections, names):
        if len(detections) == 0:
            return []

        boxes = detections.xyxy
        tracks = detections.tracker_id
        class_ids = detections.class_id

        if len(boxes) == 0:
            return []

        result = []
        for box, tracker, class_id in zip(boxes, tracks, class_ids):
            center_x = np.mean(box[[0, 2]])
            center_y = np.mean(box[[1, 3]])
            result.append((tracker, np.array([center_x, center_y]), class_id))

        return result