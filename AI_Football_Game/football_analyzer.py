# football_analyzer.py
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
from sklearn.cluster import KMeans

from detection_processor import DetectionProcessor
from pitch_processor import PitchProcessor
from visualization import Visualization
from player_cluster import PlayerCluster

class FootballAnalyzer:
    def __init__(self, ball_detection_path, detection_model_path, keypoint_model_path, video_path):
        self.ball_model = YOLO(ball_detection_path)
        self.detection_model = YOLO(detection_model_path)
        self.keypoint_model = YOLO(keypoint_model_path)
        self.video_path = video_path
        
        self.detection_processor = DetectionProcessor()
        self.pitch_processor = PitchProcessor()
        self.visualization = Visualization()
        self.player_cluster = PlayerCluster()
        
        self.tracker = sv.ByteTrack(
            lost_track_buffer=100,
            minimum_matching_threshold=0.7,
            frame_rate=30,
            minimum_consecutive_frames=3
        )
        
        self.kmeans_model = KMeans(n_clusters=2, init='k-means++', n_init=10)
        
        self.homography_matrix = None
        self.cluster_history = defaultdict(list)
        self.assigned_clusters = {}

    def run_analysis(self):
        cap = cv2.VideoCapture(self.video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)
            cv2.imshow("Football Pitch", processed_frame)
            
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        
        ball_results = self.ball_model(frame, conf=0.3, verbose=False)
        results = self.detection_model(frame, conf=0.1, verbose=False)
        filtered_results = self.detection_processor.filter_detections(results, ball_results)

        detections = sv.Detections.from_ultralytics(filtered_results[0])
        detections = self.detection_processor.smart_detection_tracking(detections, self.tracker)

        keypoints, confidences = self.detection_processor.get_keypoints_and_confidences(
            self.keypoint_model, frame
        )
        self.homography_matrix = self.pitch_processor.calculate_homography(
            keypoints, self.pitch_processor.field.vertices(), self.homography_matrix
        )
        
        centers = self.pitch_processor.get_centers_from_detections(detections, self.detection_model.names)
        cluster_labels = self.player_cluster.cluster_players(
            frame.copy(), detections, self.detection_model.names, self.kmeans_model
        )
        self.assigned_clusters = self.player_cluster.assign_clusters(
            cluster_labels, self.cluster_history, self.assigned_clusters
        )
        
        transformed_player_points = None
        if self.homography_matrix is not None:
            transformed_player_points = self.pitch_processor.transform_player_coordinates(
            centers, self.homography_matrix
        )
        
        frame_with_boxes = self.visualization.draw_boxes(
            frame.copy(), detections, self.assigned_clusters, 
            self.detection_model.names, self.visualization.class_colors
        )
        
        image = self.visualization.draw_football_pitch(
            self.pitch_processor.field.vertices(), transformed_player_points,
            self.detection_model.names, self.assigned_clusters, self.visualization.class_colors
        )
        
        return self.visualization.overlay_image(
            frame_with_boxes, image,  scale_factor=0.45, alpha=0.8
        )