
from football_analyzer import FootballAnalyzer

if __name__ == "__main__":
    analyzer = FootballAnalyzer(
        ball_detection_path = "runs/detect/train20/weights/best.pt",
        detection_model_path="runs/detect/pre-last/weights/best.pt",
        keypoint_model_path="yolo_pose/runs/pose/train10/weights/best.pt",
        video_path="epic.mp4"
    )
    analyzer.run_analysis()